import sys, os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QTabWidget, QFileDialog, QLineEdit, QSpinBox, QGraphicsView, QGraphicsScene, QInputDialog, QMessageBox, QGraphicsLineItem, QGraphicsEllipseItem
)
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtGui import QPainter, QTransform
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import ezdxf
import heapq
import numpy as np
from ezdxf.math import Vec3
import networkx as nx 
import random
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import itertools

#locates files relative to current script. if it is being run by a Pyinstaller bundle then it will retun the absolute path to hte file in the bundle 
#if not, returns the absolute path relative to the current working directory
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

#read dwg file using ezdxf
#extracts the 3d paths from the file where color code = 4, or cyan, then returns a list of paths 
#each path is a list of 3d coordinates
def plot_dwg_3d(filepath):
    doc = ezdxf.readfile(filepath)
    modelspace = doc.modelspace()

    # Get all entities in modelspace
    entities = modelspace.query('*')

    paths = []
    current_path = []
    
    #check if line or polyline 
    #if line, add start and end point to a line to the paths list 
    #if it is a polyline, creates new current_path list and adds vertices there
    #if it encounters a vertex, which probably won't happen by user input, it adds it to the current_path as well
    #since we need individual segment line weight, we really don't need to know where a complete path is.. but 
    #it is easier overtime to keep entities, like polylines, represented graphically the same as they are input in case we need to do further manipulation or debugging
    for entity in entities:
        if entity.dxftype() == 'LINE' and entity.dxf.color == 4:
            start = entity.dxf.start
            end = entity.dxf.end
            paths.append([start, end])
        elif entity.dxftype() == 'LWPOLYLINE' and entity.dxf.color == 4:
            if current_path:
                paths.append(current_path)
            current_path = [[vertex[0], vertex[1], 0] for vertex in entity.vertices()]
        elif entity.dxftype() == 'VERTEX':
            current_path.append(entity.dxf.location[:])
  
    if current_path:
        paths.append(current_path)

    return paths 

#takes a 3d vector and discards z coordinate
def vec3_to_tuple(vec3_obj):
    return (vec3_obj.x, vec3_obj.y)

#2d vectors and rounding path coordinates
#round all coordinates to 4 places to reduce memory usage and precision of coordinates
def flatten_paths(paths):
    flattened_paths = []
    for path in paths:
        flattened_path = []
        for coord in path:
            if isinstance(coord, Vec3):
                flattened_path.append(vec3_to_tuple(coord))
            else:
                flattened_path.append(tuple(round(c, 4) for c in coord))
        flattened_paths.append([(round(coord[0], 4), round(coord[1], 4)) for coord in flattened_path])
    return flattened_paths

#returns distance between two points a and b 
def euclidean_distance(a, b):
    x_diff = a[0] - b[0]
    y_diff = a[1] - b[1]
    distance = (x_diff ** 2 + y_diff ** 2) ** 0.5
    return distance

#function to determine if the point is on the path within a specific tolerance or not, just in case there could be multiple lines super super close to each other 
#uses euclidian distance to define
def is_point_on_path(point, path, tolerance=1e-6):
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]

        start_distance = euclidean_distance(point, start)
        end_distance= euclidean_distance(point, end)
        segment_length = euclidean_distance(start, end)

        if abs(start_distance + end_distance - segment_length) <= tolerance:
            return True

    return False

#splitting up polylines to have multiple segmented line weights
def split_polyline_at_intersection(polyline, intersections):
    if not intersections:
        return [polyline]

    #use euclidian distance for weights
    intersections = sorted(intersections, key=lambda x: euclidean_distance(polyline[0], x))
    split_polylines = []
    start = polyline[0]

    for intersection in intersections:
        split_polylines.append([start, intersection])
        start = intersection

    split_polylines.append([start, polyline[-1]])
    return split_polylines

def find_intersection(path1, path2):
    x1, y1 = path1[0]
    x2, y2 = path1[1]
    x3, y3 = path2[0]
    x4, y4 = path2[1]

    # Calculate the denominator
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        # The lines are parallel or coincident
        return None

    # Calculate the intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

    if 0 <= t <= 1 and 0 <= u <= 1:
        # The line segments intersect
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return (intersection_x, intersection_y)

    return None

#complex function, create weighted graph from paths of entity type line and polyline
#calls on find_interesction function, split_polyline_at_intersection
#necessary function to be able to split polyline to have individual line weight processed

def create_graph_from_paths(paths):
    G = nx.Graph()

    #initialize line segments
    line_segments = []
    for path in paths:
        for i in range(len(path) - 1):
            line_segments.append((path[i], path[i + 1]))

    #find intersections within paths and split once, hold processed to compare and only add once 
    split_segments = []
    processed_intersections = set()

    for segment1, segment2 in itertools.combinations(line_segments, 2):
        intersection = find_intersection(segment1, segment2)

        if intersection is not None and (segment1, segment2, intersection) not in processed_intersections:
            split_segments.extend(split_polyline_at_intersection(segment1, [intersection]))
            split_segments.extend(split_polyline_at_intersection(segment2, [intersection]))
            processed_intersections.add((segment1, segment2, intersection))
            processed_intersections.add((segment2, segment1, intersection))
        else:
            split_segments.append(segment1)
            split_segments.append(segment2)

    #create graph with split segments, making all lines and polylines effectively weighted edges in graph
    for segment in split_segments:
        start, end = segment

        #make sure we aren't adding self loops on nodes
        if start != end: 
            G.add_edge(start, end, weight=euclidean_distance(start, end))
            G.nodes[start]['pos'] = start
            G.nodes[end]['pos'] = end

    return G

#returns closest node in the graph to the point
def closest_point(graph, point):
    return min(graph.nodes, key=lambda node: euclidean_distance(node, point))

#begin python tool using pyqt
class HomePage(QWidget):
    def __init__(self, parent=None):
        super(HomePage, self).__init__(parent)
        layout = QVBoxLayout()

        #home page image
        image_label = QLabel(self)
        pixmap = QPixmap(resource_path('pic1.png'))
        pixmap = pixmap.scaledToWidth(400)
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label, alignment=Qt.AlignCenter)

        #welcome text
        welcome_label = QLabel("Welcome to the Factory Layout Path Optimization Tool!")
        instruction_label = QLabel("To use the Path tool, you will need 1 or more AutoCAD (.dxf) files. If your file is not in .dxf, that is okay. Simply save it as a .dxf file in AutoCAD. Next, make sure that the “walkable path” is created in your layout using the color Cyan and that no other objects in your file are of the color Cyan.")
        instruction_label.setWordWrap(True)

        #get started button
        get_started_button = QPushButton("Get Started!")
        get_started_button.clicked.connect(self.get_started)

        #add widgets to layout
        layout.addWidget(welcome_label)
        layout.addWidget(instruction_label)
        layout.addWidget(get_started_button)
        self.setLayout(layout)

    #triggers maininterface 
    def get_started(self):
        self.parent().start_main_interface()

#mainwindow 
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        #home page and tool title bar
        self.home_page = HomePage()
        self.setCentralWidget(self.home_page)
        self.setWindowTitle("Factory Layout Path Optimization Tool")

    #take user input on number of floors
    def start_main_interface(self):
        num_floors, input = QInputDialog.getInt(self, "Number of Floors", "Please enter the number of floors (between 1 and 10):", 1, 1, 10)
        if input:
            self.main_interface = MainInterface(num_floors)
            self.setCentralWidget(self.main_interface)

#visualization class, MplCanvas = Matplotlib Canvas
class MplCanvas(FigureCanvas):
        def __init__(self, parent=None, width=10, height=10, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            super(MplCanvas, self).__init__(self.fig)
            self.setParent(parent)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.updateGeometry()

        #plot the floors in 3D in plot tab
        def plot_3d_floors(self, floor_tabs, shortest_paths):
            ax = self.fig.add_subplot(111, projection='3d')

            z = 0
            z_spacing = 6

            for floor_tab in floor_tabs:
                for path in floor_tab.flat_paths_2d:
                    x_values = [coord[0] for coord in path]
                    y_values = [coord[1] for coord in path]
                    z_values = [z] * len(x_values)
                    ax.plot(x_values, y_values, z_values, marker='o', markersize=3, linestyle='-', color='blue')
                z += z_spacing

            total_length = 0

            for path_data in shortest_paths:
                floor = path_data["floor"]
                path = path_data["path"]
                x_values = [coord[0] for coord in path]
                y_values = [coord[1] for coord in path]
                z_values = [(floor - 1) * z_spacing] * len(x_values)

                ax.plot(x_values, y_values, z_values, marker='o', markersize=4, linestyle='-', color='red')

                start_point = path[0]
                end_point = path[-1]

                ax.plot([start_point[0]], [start_point[1]], [(floor - 1) * z_spacing], marker='o', markersize=8, linestyle='', color='green')
                ax.plot([end_point[0]], [end_point[1]], [(floor - 1) * z_spacing], marker='o', markersize=8, linestyle='', color='red')

                if path_data is not shortest_paths[-1]:
                    next_floor_start = shortest_paths[floor]["path"][0]
                    print(next_floor_start)
                    ax.plot([end_point[0], next_floor_start[0]], [end_point[1], next_floor_start[1]], [(floor - 1) * z_spacing, floor * z_spacing], linestyle='-', color='black')

                path_length = sum([euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1)])
                total_length += path_length

            self.fig.suptitle(f"Shortest Path Visualization\nTotal Length: {total_length:.2f} meters (Not including any stairwell lengths)", fontsize=12)

            ax.set_xlabel('Factory Length')
            ax.set_ylabel('Factory Width')
            ax.set_zlabel(f'Factory Floors')


            num_floors = len(floor_tabs)
            floor_numbers = list(range(1, num_floors + 1))
            z_values = [(floor - 1) * z_spacing for floor in floor_numbers]

            ax.set_zticks(z_values)
            ax.set_zticklabels(floor_numbers)

            self.draw()

#main interface
class MainInterface(QTabWidget):
    def __init__(self, num_floors, parent=None):
        super(MainInterface, self).__init__(parent)

        self.num_floors = num_floors
        self.start_floor = None
        self.start_point = None
        self.end_floor = None
        self.end_point = None

        self.floor_tabs = []
        for i in range(num_floors):
            tab = FloorTab(i +1)
            self.addTab(tab, f"Floor {i+1}")
            tab.start_point_selected.connect(self.set_start_point)
            tab.end_point_selected.connect(self.set_end_point)
            tab.floor_tab_updated.connect(self.check_conditions)

            self.floor_tabs.append(tab)
            
        self.floor_plan_editor = FloorPlanEditor(self.floor_tabs)
        self.canvas = MplCanvas()

        #disable calculate path 
        self.calculate_button = QPushButton("Calculate Shortest Path")
        self.calculate_button.clicked.connect(self.calculate_shortest_paths)
        self.calculate_button.setEnabled(False)
        self.setCornerWidget(self.calculate_button, Qt.TopRightCorner)

        #widget for calculate button
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_widget.setLayout(self.plot_layout)

        #add this as a tab with different text than the tabs
        self.addTab(self.plot_widget, "Plot")

    def check_conditions(self):
        #check to see if all floors have layout
        if not all(floor_tab.layout_ready for floor_tab in self.floor_tabs):
            print("Not all floors have a layout.")
            return

        #check to make sure each floor has 1 layout as long as its not the last
        if not all(len(floor_tab.stairwells) >= 1 for floor_tab in self.floor_tabs[:-1]):
            print("Not all floors have a stairwell.")
            return

        #check start and end point labeling
        if not (self.end_point):
            print("Start and end points are not labeled.")
            return

        #if these conditions met, enable button to be true
        self.calculate_button.setEnabled(True)
        
    def set_start_point(self, floor_number, point):
        current_tab = self.widget(floor_number - 1)
        point = current_tab.start_point
        self.start_floor = floor_number
        self.start_point = point
        self.disable_buttons_except(floor_number, "start")

    def set_end_point(self, floor_number, point):
        current_tab = self.widget(floor_number - 1)
        point = current_tab.end_point
        self.end_floor = floor_number
        self.end_point = point
        self.disable_buttons_except(floor_number, "end")   
        if (self.start_point and all(len(floor_tab.stairwells) >= 1 for floor_tab in self.floor_tabs[:-1])): 
            self.calculate_button.setEnabled(True)
    
    def disable_buttons_except(self, floor_number, button_type):
        for i in range(self.num_floors):
            tab = self.widget(i)
            if i != floor_number - 1:
                if button_type == "start":
                    tab.set_start_button.setEnabled(False)
                elif button_type == "end":
                    tab.set_end_button.setEnabled(False)
            else:
                if button_type == "start":
                    tab.set_start_button.setEnabled(True)
                elif button_type == "end":
                    tab.set_end_button.setEnabled(True)

    def calculate_shortest_paths(self):
        shortest_paths = self.floor_plan_editor.calculate_shortest_paths(start=self.start_point, start_floor=self.start_floor, end=self.end_point, end_floor=self.end_floor)
        self.canvas.plot_3d_floors(self.floor_tabs, shortest_paths)
        self.plot_layout.addWidget(self.canvas)

class FloorTab(QWidget):
    start_point_selected = pyqtSignal(int, QPointF)
    end_point_selected = pyqtSignal(int, QPointF)
    total_floors = 0
    floor_tab_updated = pyqtSignal()

    def __init__(self, floor_number, parent=None):

        super(FloorTab, self).__init__(parent)
        self.flat_paths_2d = None
        self.floor_number = floor_number
        self.layout_uploaded = False  # Add this line

        #add a QGraphicsView widget to display the QGraphicsScene
        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.graphics_view.setTransform(QTransform.fromScale(1, -1))
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.graphics_view.mousePressEvent = self.graphics_view_mouse_press_event
        self.graphics_view.mouseMoveEvent = self.graphics_view_mouse_move_event

        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)

        self.browse_button = QPushButton("Browse for the layout", self)
        self.browse_button.clicked.connect(self.browse_button_clicked)

        self.set_stairwells_button = QPushButton("Toggle Set Stairwells (Press Again When Done)", self)
        self.set_stairwells_button.clicked.connect(self.set_stairwells_button_clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(self.graphics_view)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.set_stairwells_button)

        self.stairwells = []
        self.setting_stairwells = False
        self.stair_icon = None

        #START AND END POINT
        self.set_start_button = QPushButton("Toggle Set Start Point (Press Again When Done)", self)
        self.set_start_button.clicked.connect(self.set_start_button_clicked)

        self.set_end_button = QPushButton("Toggle Set End Point (Press Again When Done)", self)
        self.set_end_button.clicked.connect(self.set_end_button_clicked)

        layout.addWidget(self.set_start_button)
        layout.addWidget(self.set_end_button)

        self.start_point = None
        self.start_point_icon = None
        self.setting_start_point = False

        self.end_point = None
        self.end_point_icon = None
        self.setting_end_point = False

    @property
    def layout_ready(self):
        return self.layout_uploaded

    def set_start_button_clicked(self):
        self.setting_start_point = not self.setting_start_point
        if self.setting_start_point and self.start_point is not None:
            self.start_point_selected.emit(self.floor_number, self.start_point)
    
    def set_end_button_clicked(self):
        self.setting_end_point = not self.setting_end_point
        if self.setting_end_point and self.end_point is not None:
            self.end_point_selected.emit(self.floor_number, self.end_point)

    def set_start_point(self, point, floor_number):
        if self.start_point_icon is not None:
            self.graphics_scene.removeItem(self.start_point_icon)
            self.start_point_icon = None

        start_point_color = QColor(0, 255, 0)
        start_point_radius = 0.5
        self.start_point = point
        print("Starting point SET: ", point)
        self.start_point_icon = QGraphicsEllipseItem(QRectF(point.x() - start_point_radius, point.y() - start_point_radius, 2 * start_point_radius, 2 * start_point_radius))
        
        #set the brush and pen color to green
        brush = QBrush(start_point_color)
        pen = QPen(start_point_color, 0)
        
        self.start_point_icon.setBrush(brush)
        self.start_point_icon.setPen(pen)
        self.graphics_scene.addItem(self.start_point_icon)

        self.start_floor = floor_number
        self.start_point = point
        self.floor_tab_updated.emit()

    def set_end_point(self, point, floor_number):
        if self.end_point_icon is not None:
            self.graphics_scene.removeItem(self.end_point_icon)
            self.end_point_icon = None

        end_point_color = QColor(255, 0, 0)
        end_point_radius = 0.5
        self.end_point = point
        
        print("End point SET: ", point)
        self.end_point_icon = QGraphicsEllipseItem(QRectF(point.x() - end_point_radius, point.y() - end_point_radius, 2 * end_point_radius, 2 * end_point_radius))
        
        #set the brush and pen color to red
        brush = QBrush(end_point_color)
        pen = QPen(end_point_color, 0) 
        
        self.end_point_icon.setBrush(brush)
        self.end_point_icon.setPen(pen) 
        self.graphics_scene.addItem(self.end_point_icon)

        self.end_floor = floor_number
        self.end_point = point
        self.floor_tab_updated.emit() 

    def find_closest_point_on_paths(self, point, paths):
        closest_distance = float('inf')
        closest_point = None

        for path in paths:
            for node in path:
                #calculate the distance between the point and the node
                distance = euclidean_distance(point, node)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = node

        return closest_point

    def closest_stairwell(self, point):
        closest_distance = float('inf')
        closest_stairwell = None

        for stairwell in self.stairwells:
            distance = euclidean_distance(point, (stairwell["point"].x(), stairwell["point"].y()))

            if distance < closest_distance:
                closest_distance = distance
                closest_stairwell = stairwell

        return closest_stairwell

    @staticmethod
    def plot_graph_with_weights(graph):
        pos = nx.get_node_attributes(graph, 'pos')
        edge_labels = {(u, v): round(d['weight'], 2) for u, v, d in graph.edges(data=True)}

        plt.figure(figsize=(12, 12))
        nx.draw(graph, pos, with_labels=False, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
        plt.show()

    def shortest_path(self, graph, start, end):

        print("---SHORTEST PATH CALCULATION----")
        print("Graph: ",graph)
        print("Start: ", start)
        print("End: ", end)
        
        try:
            #FloorTab.plot_graph_with_weights(graph)
            return nx.dijkstra_path(graph, start, end, weight = 'weight')
        except nx.NetworkXNoPath:
            print("No path found!")
            FloorTab.plot_graph_with_weights(graph)
            return None

    def graphics_view_mouse_press_event(self, event):
        if event.button() == Qt.LeftButton and self.setting_stairwells:
            point = self.graphics_view.mapToScene(event.pos())
            flat_paths_2d = self.flat_paths_2d
            self.add_stairwell(point)

        if event.button() == Qt.LeftButton and not self.setting_stairwells:
            point = self.graphics_view.mapToScene(event.pos()).toPoint()
            flat_paths_2d = self.flat_paths_2d
            closest_point = self.find_closest_point_on_paths((point.x(), point.y()), flat_paths_2d)

            if self.setting_start_point:
                self.set_start_point(QPointF(closest_point[0], closest_point[1]), self.floor_number)
                self.start_point_selected.emit(self.floor_number, QPointF(closest_point[0], closest_point[1]))

            elif self.setting_end_point:
                self.set_end_point(QPointF(closest_point[0], closest_point[1]), self.floor_number)
                self.end_point_selected.emit(self.floor_number, QPointF(closest_point[0], closest_point[1]))

    def graphics_view_mouse_move_event(self, event):
        if self.setting_stairwells:
            point = self.graphics_view.mapToScene(event.pos())
            if self.stair_icon is None:
                stair_icon_path = resource_path("stair_icon.png")
                pixmap = QPixmap(stair_icon_path)
                
                #scale the QPixmap to a height of 3 while maintaining the aspect ratio
                scaled_pixmap = pixmap.scaledToHeight(3, Qt.SmoothTransformation)
                
                self.stair_icon = self.graphics_scene.addPixmap(scaled_pixmap)
            self.stair_icon.setPos(point.x() - self.stair_icon.pixmap().width() / 2, point.y() - self.stair_icon.pixmap().height() / 2)
 
    def add_stairwell(self, point):
        stair_icon_path = resource_path("stair_icon.png")
        pixmap = QPixmap(stair_icon_path)
        
        #scale the QPixmap to a height of 3 while maintaining the aspect ratio
        scaled_pixmap = pixmap.scaledToHeight(3, Qt.SmoothTransformation)

        flat_paths_2d = self.flat_paths_2d
        closest_point = self.find_closest_point_on_paths((point.x(), point.y()), flat_paths_2d)

        stair_icon_item = self.graphics_scene.addPixmap(scaled_pixmap)
        stair_icon_item.setPos(closest_point[0] - scaled_pixmap.width() / 2, closest_point[1] - scaled_pixmap.height() / 2)
        print("Stairwell SET: ", QPointF(closest_point[0], closest_point[1]))
        self.stairwells.append({"point": QPointF(closest_point[0], closest_point[1]), "icon": stair_icon_item})
        self.floor_tab_updated.emit()

    def set_stairwells_button_clicked(self):
        self.setting_stairwells = not self.setting_stairwells
        if not self.setting_stairwells and self.stair_icon is not None:
            self.graphics_scene.removeItem(self.stair_icon)
            self.stair_icon = None

    def browse_button_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "DXF Files (*.dxf)")

        if file_path:
            self.flat_paths_2d = flatten_paths(plot_dwg_3d(file_path))
            print(self.flat_paths_2d)
            self.draw_paths(self.flat_paths_2d)
        
        self.layout_uploaded = True
        self.floor_tab_updated.emit()

    def fit_view_to_content(self):
        self.graphics_view.setSceneRect(self.graphics_scene.itemsBoundingRect())
        self.graphics_view.fitInView(self.graphics_view.sceneRect(), Qt.KeepAspectRatio)

    def draw_paths(self, flat_paths_2d):

        self.graphics_scene.clear()
        #parameters for the appearance of the path lines and vertices
        path_line_width = .3
        path_line_color = QColor(0, 255, 255)
        vertex_radius = .1
        vertex_color = QColor(0, 0, 255)

        for path in flat_paths_2d:
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                line = QGraphicsLineItem(start[0], start[1], end[0], end[1])
                pen = QPen(path_line_color, path_line_width)
                line.setPen(pen)
                self.graphics_scene.addItem(line)

                #draw start and end points as small circles
                start_circle = QGraphicsEllipseItem(QRectF(start[0] - vertex_radius, start[1] - vertex_radius, 2 * vertex_radius, 2 * vertex_radius))
                start_circle.setBrush(QBrush(vertex_color))
                self.graphics_scene.addItem(start_circle)

                end_circle = QGraphicsEllipseItem(QRectF(end[0] - vertex_radius, end[1] - vertex_radius, 2 * vertex_radius, 2 * vertex_radius))
                end_circle.setBrush(QBrush(vertex_color))
                self.graphics_scene.addItem(end_circle)

        self.fit_view_to_content()

class FloorPlanEditor:
    def __init__(self, floor_tabs):
        self.floor_tabs = floor_tabs

        #round coordinates of all points in flat_paths_2d of each floor tab
        for floor_tab in self.floor_tabs:
            if floor_tab.flat_paths_2d:
                floor_tab.flat_paths_2d = (floor_tab.flat_paths_2d)

    def calculate_shortest_paths(self, start, start_floor, end, end_floor):
        current_floor = start_floor
        current_point = (round(start.x(), 4), round(start.y(), 4))
        end_point = (round(end.x(), 4), round(end.y(), 4))

        shortest_paths = []
        print("Starting floor: ", current_floor)
        print("Starting point: ", current_point)
        print("End floor: ", end_floor)
        print("End point: ", end_point)
        
        while current_floor != end_floor:
            floor_tab = self.floor_tabs[current_floor - 1]

            #find the nearest stairwell
            closest_stairwell = floor_tab.closest_stairwell(current_point)
            closest_stairwell_point = (closest_stairwell["point"].x(), closest_stairwell["point"].y())
            print("Closest stairwell: ", closest_stairwell_point)

            graph = create_graph_from_paths(floor_tab.flat_paths_2d)
            shortest_path = floor_tab.shortest_path(graph, current_point, closest_stairwell_point)

            if shortest_path:
                shortest_paths.append({"floor": current_floor, "path": shortest_path})
                print("shortest path: ", shortest_path)

            current_point = closest_stairwell_point
            print("Current point", current_point)

            #find the closest point on the path to start again
            closest_point = floor_tab.find_closest_point_on_paths(current_point, self.floor_tabs[current_floor].flat_paths_2d)
            current_point = closest_point

            #move to the next floor
            current_floor += 1

    #if we have reached the target floor
        if current_floor == end_floor:
            floor_tab = self.floor_tabs[end_floor - 1]
            graph = create_graph_from_paths(floor_tab.flat_paths_2d)

            for path in floor_tab.flat_paths_2d:
                x_values = [coord[0] for coord in path]
                y_values = [coord[1] for coord in path]
                plt.plot(x_values, y_values, marker='o', markersize=4, linestyle='-', color='cyan')

            shortest_path = floor_tab.shortest_path(graph, current_point, end_point)
            if shortest_path:
                shortest_paths.append({"floor": end_floor, "path": shortest_path})
                print("shortest path: ", shortest_path)
            
        return shortest_paths

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())




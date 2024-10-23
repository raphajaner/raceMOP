import numpy as np
import pyglet
import yaml
from PIL import Image
from pyglet.gl import *
from pyglet.window import key

from f110_gym.envs.env_utils import get_vertices, get_trmtx

# zooming constants for mouse scroll
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.51
CAR_WIDTH = 0.27

MAX_SPEED = 8
MAX_STEERING = 0.42
MAX_BAR_LENGTH = 100
BAR_START_X = 10
BAR_START_Y = 10
BAR_HEIGHT = 18


class TextDisplay:
    def __init__(self, window, text, x, y, font_size=18, color=(127, 127, 127, 127), bold=True):
        self.label = pyglet.text.Label(
            text, font_size=font_size, x=x, y=y, anchor_x='left', anchor_y='center', color=color, bold=bold)
        self.window = window

    @property
    def text(self):
        return self.label.text

    @text.setter
    def text(self, value):
        self.label.text = value

    def draw(self):
        """Draw the label.

        The OpenGL state is assumed to be at default values, except
        that the MODELVIEW and PROJECTION matrices are ignored.  At
        the return of this method the matrix mode will be MODELVIEW.
        """
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, self.window.width, 0, self.window.height, -1, 1)

        self.label.draw()

        gl.glPopMatrix()

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()

    # def _hook_flip(self):
    #     self._window_flip()


class Bar01Display:
    def __init__(self, window, text, x, y, width, height, font_size=18, color=(127, 127, 127, 127), bold=True):

        self.label = pyglet.text.Label(
            text, font_size=font_size, x=x, y=y, anchor_x='left', anchor_y='center', color=color, bold=bold)

        self.x = x + 150
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.window = window

        self._value = 0.0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        # self.draw()

    def _draw_bar(self, value):

        bar_length = value * MAX_BAR_LENGTH
        if value > 1.0:
            # make gray if the bar is full
            bar_length = MAX_BAR_LENGTH
            glColor4ub(200, 200, 200, 255)
        else:
            # make blue if the bar is not full
            glColor4ub(0, 255, 0, 255)

        glBegin(GL_QUADS)
        glVertex2f(self.x, self.y + self.height / 2)
        glVertex2f(self.x + bar_length, self.y + self.height / 2)
        glVertex2f(self.x + bar_length, self.y - self.height / 2)
        glVertex2f(self.x, self.y - self.height / 2)
        glEnd()

        # add ticks to the bar
        glColor4ub(0, 0, 0, 255)
        for i in range(0, MAX_BAR_LENGTH + 1, 25):
            glBegin(GL_LINES)
            glVertex2f(self.x + i, self.y - self.height / 2 - 2)
            glVertex2f(self.x + i, self.y + self.height / 2 + 2)
            glEnd()

    def draw(self):
        """Draw the label.

        The OpenGL state is assumed to be at default values, except
        that the MODELVIEW and PROJECTION matrices are ignored.  At
        the return of this method the matrix mode will be MODELVIEW.
        """
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, self.window.width, 0, self.window.height, -1, 1)

        self._draw_bar(2.0)
        self._draw_bar(self._value)
        self.label.draw()

        gl.glPopMatrix()

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()


class Bar11Display(Bar01Display):
    def __init__(self, window, text, x, y, width, height, font_size=18, color=(127, 127, 127, 127), bold=True):

        self.label = pyglet.text.Label(text, font_size=font_size, x=x, y=y, anchor_x='left', anchor_y='center',
                                       color=color, bold=bold)

        self.x = x + 150

        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.window = window

        self._value = 0.0

    def _draw_bar(self, value):
        bar_length = value * MAX_BAR_LENGTH
        if value > 1.0:
            # make gray if the bar is full
            glColor4ub(200, 200, 200, 255)
            glBegin(GL_QUADS)
            glVertex2f(self.x, self.y + self.height / 2)
            glVertex2f(self.x + MAX_BAR_LENGTH, self.y + self.height / 2)
            glVertex2f(self.x + MAX_BAR_LENGTH, self.y - self.height / 2)
            glVertex2f(self.x, self.y - self.height / 2)
            glEnd()
        else:
            # make green if the bar is not full
            glColor4ub(0, 255, 0, 255)
            glBegin(GL_QUADS)
            glVertex2f(self.x + MAX_BAR_LENGTH / 2, self.y + self.height / 2)
            glVertex2f(self.x + MAX_BAR_LENGTH / 2 + bar_length / 2, self.y + self.height / 2)
            glVertex2f(self.x + MAX_BAR_LENGTH / 2 + bar_length / 2, self.y - self.height / 2)
            glVertex2f(self.x + MAX_BAR_LENGTH / 2, self.y - self.height / 2)
            glEnd()

        glColor4f(0, 0, 0, 255)
        for i in range(0, MAX_BAR_LENGTH + 1, 25):
            glBegin(GL_LINES)
            glVertex2f(self.x + i, self.y - self.height / 2 - 2)
            glVertex2f(self.x + i, self.y + self.height / 2 + 2)
            glEnd()


class EnvRenderer(pyglet.window.Window):
    """A window class inherited from pyglet.window.Window,

    Handles the camera/projection interaction, resizing window, and rendering the environment
    """

    def __init__(self, width, height, *args, **kwargs):
        """Initialize the window with the given width and height

        Args:
            width (int): width of the window in pixels
            height (int): height of the window in pixels
        """

        self.map_name = 'Unknown'
        conf = Config(
            sample_buffers=1,
            samples=4,
            depth_size=16,
            double_buffer=True
        )
        super().__init__(width, height, caption=f'F1TENTH Autonomous Racing', config=conf, resizable=True, vsync=False,
                         *args,
                         **kwargs)

        # initialize camera values
        self.left = -width / 2
        self.right = width / 2
        self.bottom = -height / 2
        self.top = height / 2
        self.zoom_level = 1.0
        self.zoomed_width = width
        self.zoomed_height = height

        # current batch that keeps track of all graphics
        self.batch = pyglet.graphics.Batch()

        # current env map
        self.map_points = None
        self.map_verts = None
        # current env agents.py poses, (num_agents, 3), columns are (x, y, theta)
        self.poses = None

        # current env agents.py vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        self.vertices = None
        self.lidar_points = None

        self.track_name_display = TextDisplay(
            self, 'Track: Unknown', 10, self.height - 10, font_size=18, color=(127, 127, 127, 127), bold=True)

        self.lap_time_display = TextDisplay(
            self, 'Lap Time: 0.0', 10, self.height - 40, font_size=18, color=(127, 127, 127, 127), bold=True)

        self.lap_count_display = TextDisplay(
            self, '(0/2)', 235, self.height - 40, font_size=18, color=(127, 127, 127, 127), bold=True)

        # self.vel_display = TextDisplay(
        #     self, 'Ego Vel: 0', 10, self.height - 70, font_size=18, color=(127, 127, 127, 127), bold=True)

        self.vel_bar = Bar01Display(self, 'Action v_t:', 10, self.height - 80, MAX_BAR_LENGTH, BAR_HEIGHT)
        self.steering_bar = Bar11Display(self, 'Action \u03B4_t:', 10, self.height - 110, MAX_BAR_LENGTH, BAR_HEIGHT)

        # make sure that the bar in in the window
        self.BAR_START_X = self.width - 200
        self.BAR_START_Y = self.height - 10

        self.fps_display = pyglet.window.FPSDisplay(self)
        self.fps_display.label.font_size = 18

        self.frame = 0
        self.ego_idx = 0

        self.paused = False

    def on_key_press(self, symbol, modifiers):
        if symbol == key.P:
            self.paused = not self.paused
            if self.paused:
                while self.paused:
                    self.dispatch_events()

    def update_map(self, map_path, map_ext, map_name):
        """ Update the map being drawn by the renderer.

        Converts image to a list of 3D points representing each obstacle pixel in the map.

        Args:
            map_path (str): absolute path to the map without extensions
            map_ext (str): extension for the map image file
        """
        self.map_name = map_name

        # load map metadata
        with open(map_path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata['resolution']
                origin = map_metadata['origin']
                origin_x = origin[0]
                origin_y = origin[1]
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        map_img = np.array(Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)

        ###########
        rescale_factor = 1
        import PIL
        import cv2

        map_img = PIL.Image.fromarray(map_img)

        image = map_img.convert('L')
        image = image.resize((image.width * rescale_factor, image.height * rescale_factor), Image.NEAREST)

        image = np.array(image) / 255.0

        #
        image[image > 0.1] = 1.0
        image[image <= 0.1] = 0.0

        if True:  # Convert the matrix to an image
            image = np.uint8(image * 255)
            # Find contours
            contours, _ = cv2.findContours(image, 3, cv2.CHAIN_APPROX_NONE)

            # Initialize a label matrix
            label_matrix = np.zeros_like(image)

            # Label each area
            label = 1
            for contour in contours:
                # Fill the area inside the contour
                cv2.drawContours(label_matrix, [contour], -1, color=label, thickness=cv2.FILLED)
                label += 1

            segments = []
            for i in range(1, label):
                # Create a mask for the current area
                mask = np.where(label_matrix == i, 255, 0).astype(np.uint8)
                # Extract the area
                segmented_area = cv2.bitwise_and(image, image, mask=mask)
                area = np.sum(segmented_area) / 255.0
                segments.append((area, segmented_area))

            # sort the segments by area
            segments.sort(key=lambda x: x[0], reverse=True)

            # the third area is the track (most certain)
            image[segments[2][1] > 0] = 255 * 0.9
            image = image / 255.0

        map_img = np.array(image)

        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        # convert map pixels to coordinates
        range_x = np.arange(map_width)
        range_y = np.arange(map_height)
        map_x, map_y = np.meshgrid(range_x, range_y)

        # interpolate the meshgrid to increase the resolution
        map_x = map_x.flatten()
        map_y = map_y.flatten()

        map_x = (map_x * map_resolution + origin_x).flatten()
        map_y = (map_y * map_resolution + origin_y).flatten()
        map_z = np.zeros(map_y.shape)
        map_coords = np.vstack((map_x, map_y, map_z))

        map_img = map_img * 255

        map_mask = map_img < 10.  # that's the threshold for the obstacle
        map_mask_flat = map_mask.flatten()
        map_points = 50. * map_coords[:, map_mask_flat].T

        # map_points = 50. * map_coords.T

        if self.map_verts is not None:
            for v in self.map_verts:
                v.delete()

        self.map_verts = []
        glPointSize(7.0)  # Set this to the desired point size
        # glPointSize(5.0)  # Set this to the desired point size
        for i in range(map_points.shape[0]):
            map_verts = self.batch.add(
                1, GL_POINTS, None,
                ('v3f/stream', [map_points[i, 0], map_points[i, 1], map_points[i, 2]]),
                # ('c3B/stream', [255, 255, 255])
                # ('c3B/stream', [183, 193, 222])
                ('c4B/stream', [0, 0, 0, 255])  # Set wall color to black

            )
            self.map_verts.append(map_verts)
        self.map_points = map_points

        # also plot the contours of the track (that is in gray) where the image is  > 10 nut < 20

        contour_mask = (map_img > 10) & (map_img < 240)

        contour_mask_flat = contour_mask.flatten()
        contour = contour.astype(np.float64)
        contour = contour.flatten()
        contour_points = 50. * map_coords[:, contour_mask_flat].T
        for i in range(contour_points.shape[0]):
            map_verts = self.batch.add(
                1, GL_POINTS, None,
                ('v3f/stream', [contour_points[i, 0], contour_points[i, 1], contour_points[i, 2]]),
                # ('c3B/stream', [255, 255, 255])
                # ('c3B/stream', [183, 193, 222])
                ('c4B/stream', [183, 193, 222, 255])  # Set wall color to black

            )
            self.map_verts.append(map_verts)

    def on_resize(self, width, height):
        """Callback function on window resize, overrides inherited method, and updates camera values.

        Args:
            width (int): new width of window
            height (int): new height of window
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        (width, height) = self.get_size()
        self.left = -self.zoom_level * width / 2
        self.right = self.zoom_level * width / 2
        self.bottom = -self.zoom_level * height / 2
        self.top = self.zoom_level * height / 2
        self.zoomed_width = self.zoom_level * width
        self.zoomed_height = self.zoom_level * height

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Callback function on mouse drag, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (int): Relative X position from the previous mouse position.
            dy (int): Relative Y position from the previous mouse position.
            buttons (int): Bitwise combination of the mouse buttons currently pressed.
            modifiers (int): Bitwise combination of any keyboard modifiers currently active.
        """

        # pan camera
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top -= dy * self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        """Callback function on mouse scroll, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            scroll_x (float): Amount of movement on the horizontal axis.
            scroll_y (float): Amount of movement on the vertical axis.
        """
        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1

        # If zoom_level is in the proper range
        if .01 < self.zoom_level * f < 10:
            self.zoom_level *= f

            (width, height) = self.get_size()

            mouse_x = x / width
            mouse_y = y / height

            mouse_x_in_world = self.left + mouse_x * self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y * self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

    def on_close(self):
        """Callback function when the 'x' is clicked on the window, overrides inherited method.

        Also throws exception to end the python program when in a loop.
        """
        super().on_close()
        # out.release()

        raise Exception('Rendering window was closed.')

    def on_draw(self):
        """ Function when the pyglet is drawing.



        The function draws the batch created that includes the map points, the agents.py polygons, and the information
        text, and the fps display.
        """
        # if map and poses doesn't exist, raise exception
        if self.map_points is None:
            raise Exception('Map not set for renderer.')
        if self.poses is None:
            raise Exception('Agent poses not updated for renderer.')

        # Initialize Projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()

        # Clear window with ClearColor
        glClearColor(1.0, 1.0, 1.0, 1.0)  # Set background color to white
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set orthographic projection matrix
        glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        if False:
            # Translate to the vehicle position
            glTranslatef(self.poses[0, 0] * 50, self.poses[0, 1] * 50, 0)
            # Rotate the window
            glRotatef(-self.orientation * 180 / np.pi + 90, 0, 0, 1)
            # Translate back to the original position
            glTranslatef(-self.poses[0, 0] * 50, -self.poses[0, 1] * 50, 0)

        # rotate the map so that the vehicle orientation is aligned with the x-axis using self.orientation, rotate
        # around center of the window
        # glTranslatef(-self.poses[0, 0], -self.poses[0, 1], 0)

        # Draw all batches
        self.batch.draw()

        # rotate the window so that the vehicle orientation is aligned with the x-axis using self.orientation, rotate
        # around center of the window

        glPopMatrix()

        self.fps_display.draw()

        self.track_name_display.draw()
        self.lap_time_display.draw()
        self.lap_count_display.draw()
        # self.vel_display.draw()
        self.vel_bar.draw()
        self.steering_bar.draw()

        # Capture current frame
        # pyglet_image = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        # buffer = pyglet_image.get_data('RGB', pyglet_image.width * 3)
        # frame = np.frombuffer(buffer, dtype=np.uint8).reshape(pyglet_image.height, pyglet_image.width, 3)
        # frame = cv2.flip(frame, 0)  # Flip vertically
        #
        # # Write frame to video file
        # out.write(frame)

        # out  = pyglet.image.get_buffer_manager().get_color_buffer().save('out/frame' + str(self.frame) + '.png')
        self.frame += 1

        # import pdb
        # pdb.set_trace()

    def update_obs(self, obs):
        """Updates the renderer with the latest observation from the gym environment.

        Args:
            obs (dict): observation dict from the gym env
        """

        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        poses_theta = obs['poses_theta']

        num_agents = len(poses_x)
        if self.poses is None:
            self.cars = []
            for i in range(num_agents):
                if i == self.ego_idx:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices),
                                         # ('c3B', [172, 97, 185, 172, 97, 185, 172, 97, 185, 172, 97, 185]))
                                         ('c3B', [0, 0, 200, 0, 0, 200, 0, 0, 200, 0, 0, 200]))
                    self.cars.append(car)
                else:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices),
                                         ('c3B', [200, 0, 0, 200, 0, 0, 200, 0, 0, 200, 0, 0]))
                    # ('c3B', [99, 52, 94, 99, 52, 94, 99, 52, 94, 99, 52, 94]))
                    self.cars.append(car)

        # draw the lidar signals points in the obs to the map
        # import pdb
        # pdb.set_trace()

        if False:
            lidar_distances = obs['aaa_scans'][self.ego_idx]
            fov = np.linspace(-np.deg2rad(135), np.deg2rad(135), 1080)

            # get the xy pairs
            lidar_points = np.zeros((len(lidar_distances), 2))
            H = get_trmtx(np.array([poses_x[self.ego_idx], poses_y[self.ego_idx], poses_theta[self.ego_idx]]))

            for i in range(len(lidar_distances)):
                # transform them to the global frame of reference of this window
                point_local = np.array([lidar_distances[i] * np.cos(fov[i]), lidar_distances[i] * np.sin(fov[i]), 0, 1])
                point = H.dot(point_local)
                lidar_points[i, 0] = point[0]
                lidar_points[i, 1] = point[1]

            if self.lidar_points is not None:
                for v in self.lidar_points:
                    v.delete()

            self.lidar_points = []

            glPointSize(7.0)
            # import pdb
            # pdb.set_trace()
            for i in range(lidar_points.shape[0]):
                lidar_verts = self.batch.add(
                    1, GL_POINTS, None,
                    ('v3f/stream', [lidar_points[i, 0] * 50, lidar_points[i, 1] * 50, 0]),
                    ('c4B/stream', [200, 200, 0, 255])
                )
                self.lidar_points.append(lidar_verts)

        poses = np.stack((poses_x, poses_y, poses_theta)).T
        for j in range(poses.shape[0]):
            vertices_np = 50. * get_vertices(poses[j, :], CAR_LENGTH, CAR_WIDTH)
            vertices = list(vertices_np.flatten())
            self.cars[j].vertices = vertices
        self.poses = poses

        self.track_name_display.text = f'Track: {self.map_name}'
        # self.lap_time_display.text = f'Lap Time: {obs["lap_times"][self.ego_idx]:02.02f}'
        lap_time = obs["lap_times"][self.ego_idx]
        lap_time_str = f"{lap_time:.2f}"
        if lap_time < 10:
            lap_time_str = "0" + lap_time_str
        self.lap_time_display.text = f'Lap Time: {lap_time_str}'
        self.lap_count_display.text = f'({obs["lap_counts"][self.ego_idx]:.0f}/2)'
        # self.vel_display.text = f'Ego Vel: {obs["linear_vels_x"][self.ego_idx]:.0f}'
        self.vel_bar.value = obs["prev_action"][self.ego_idx][1] / MAX_SPEED
        self.steering_bar.value = - obs["prev_action"][self.ego_idx][0] / MAX_STEERING

        if abs(obs["prev_action"][self.ego_idx][0]) > MAX_STEERING:
            print('Steering angle is greater than the max steering angle')
            # import pdb
            # pdb.set_trace()

        # add a bar plot from -1 to 1 that shows the steering angle of the ego car not using the batch

        # set the camera to the ego car
        x = self.cars[0].vertices[::2]
        y = self.cars[0].vertices[1::2]
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_range = self.right - self.left
        y_range = self.top - self.bottom
        # x_range = max(x_range, y_range)
        # y_range = max(x_range, y_range)
        self.left = x_center - x_range / 2
        self.right = x_center + x_range / 2
        self.bottom = y_center - y_range / 2
        self.top = y_center + y_range / 2
        self.zoomed_width = x_range
        self.zoomed_height = y_range
        self.orientation = poses[0, 2]

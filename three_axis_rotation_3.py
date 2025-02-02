""" Three-axis rotation """
import numpy as np
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import ttk
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import proj3d

""" Global variables """

""" Animation control """
is_play = False

""" Axis vectors """
vector_x_axis = np.array([1., 0., 0.])
vector_y_axis = np.array([0., 1., 0.])
vector_z_axis = np.array([0., 0., 1.])

""" Other parameters """
rot_velocity_a, rot_velocity_b, rot_velocity_c = 1., 1., 1.
is_normalized = True
is_rot_roll_pitch_yaw = True
is_wave = False
is_path_a_on = True
is_path_b_on = True
is_path_c_on = True
is_path_r_on = True

angle_step_deg = 1.
angle_acc_deg = 0.

wave_number_a = 1.
wave_number_b = 1.
wave_number_c = 1.

wave_phase_a = 0.
wave_phase_b = 0.
wave_phase_c = 0.

wave_a = 0.
wave_b = 0.
wave_c = 0.

adjustment = 1. / np.pi

""" Create figure and axes """
title_ax0 = "Three-axis rotation"
title_tk = title_ax0

x_min = -2.
x_max = 2.
y_min = -2.
y_max = 2.
z_min = -2.
z_max = 2.

fig = Figure()
ax0 = fig.add_subplot(121, projection='3d')
ax0.set_box_aspect((1, 1, 1))
ax0.grid()
ax0.set_title(title_ax0)
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")
ax0.set_xlim(x_min, x_max)
ax0.set_ylim(y_min, y_max)
ax0.set_zlim(z_min, z_max)

ax1 = fig.add_subplot(122)
ax1.set_title("Wave of rotation velocity")
ax1.set_xlabel("Phase")
ax1.set_ylabel("Velocity")
ax1.set_xlim(0., 360,)
ax1.set_ylim(-4., 4.)
# ax1.set_aspect("equal")
ax1.set_aspect(30)
ax1.grid()

""" Embed in Tkinter """
root = tk.Tk()
root.title(title_tk)
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(expand=True, fill="both")

toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()

""" Global objects of Tkinter """
var_axis_op = tk.IntVar()
var_turn_op = tk.IntVar()
var_apply_v = tk.IntVar(root)

""" Classes and functions """


class Counter:
    def __init__(self, is3d=None, ax=None, xy=None, z=None, label=""):
        self.is3d = is3d if is3d is not None else False
        self.ax = ax
        self.x, self.y = xy[0], xy[1]
        self.z = z if z is not None else 0
        self.label = label

        self.count = 0

        if not is3d:
            self.txt_step = self.ax.text(self.x, self.y, self.label + str(self.count))
        else:
            self.txt_step = self.ax.text2D(self.x, self.y, self.label + str(self.count))
            self.xz, self.yz, _ = proj3d.proj_transform(self.x, self.y, self.z, self.ax.get_proj())
            self.txt_step.set_position((self.xz, self.yz))

    def count_up(self):
        self.count += 1
        self.txt_step.set_text(self.label + str(self.count))

    def reset(self):
        self.count = 0
        self.txt_step.set_text(self.label + str(self.count))

    def get(self):
        return self.count


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


class ThreeArrow:
    def __init__(self, ax=None, xyz=None, direction=None):
        self.ax = ax
        self.xyz = xyz
        self.direction = direction
        self.is_rot_roll_pitch_yaw = True
        # self.size = size
        # self.color = color

        self.roll_axis = np.array([1., 0., 0.])
        self.pitch_axis = np.array([0., 1., 0.])
        self.yaw_axis = np.array([0., 0., 1.])

        self.qvr_roll_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                            self.roll_axis[0], self.roll_axis[1], self.roll_axis[2],
                                            length=1, color="red", normalize=True, linewidth=2)

        self.qvr_pitch_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                             self.pitch_axis[0], self.pitch_axis[1], self.pitch_axis[2],
                                             length=1, color="blue", normalize=True, linewidth=2)

        self.qvr_yaw_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.yaw_axis[0], self.yaw_axis[1], self.yaw_axis[2],
                                           length=1, color="green", normalize=True, linewidth=2)

        if self.is_rot_roll_pitch_yaw:
            self.vector_a = self.roll_axis * rot_velocity_a
            self.vector_b = self.pitch_axis * rot_velocity_b
            self.vector_c = self.yaw_axis * rot_velocity_c
        else:
            self.vector_a = vector_x_axis * rot_velocity_a
            self.vector_b = vector_y_axis * rot_velocity_b
            self.vector_c = vector_z_axis * rot_velocity_c

        self.qvr_a_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.vector_a[0], self.vector_a[1], self.vector_a[2],
                                         length=1, color="red", linestyle="-.", linewidth=1)

        self.qvr_b_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.vector_b[0], self.vector_b[1], self.vector_b[2],
                                         length=1, color="blue", linestyle="-.", linewidth=1)

        self.qvr_c_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.vector_c[0], self.vector_c[1], self.vector_c[2],
                                         length=1, color="green", linestyle="-.", linewidth=1)

        self.vector_resultant = self.vector_a + self.vector_b + self.vector_c

        self.qvr_resultant_vector = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                                   self.vector_resultant[0], self.vector_resultant[1],
                                                   self.vector_resultant[2], length=1, color="black",
                                                   normalize=False, linewidth=1, linestyle="-.")

        self.is_path_a_on = True
        self.is_path_b_on = True
        self.is_path_c_on = True
        self.is_path_r_on = True

        self.x_path_roll_axis = []
        self.y_path_roll_axis = []
        self.z_path_roll_axis = []
        self.path_roll_axis, = self.ax.plot(np.array(self.x_path_roll_axis),
                                            np.array(self.y_path_roll_axis),
                                            np.array(self.z_path_roll_axis),
                                            color="red", linewidth=0.5)

        self.x_path_pitch_axis = []
        self.y_path_pitch_axis = []
        self.z_path_pitch_axis = []
        self.path_pitch_axis, = self.ax.plot(np.array(self.x_path_pitch_axis),
                                             np.array(self.y_path_pitch_axis),
                                             np.array(self.z_path_pitch_axis),
                                             color="blue", linewidth=0.5)

        self.x_path_yaw_axis = []
        self.y_path_yaw_axis = []
        self.z_path_yaw_axis = []
        self.path_yaw_axis, = self.ax.plot(np.array(self.x_path_yaw_axis),
                                           np.array(self.y_path_yaw_axis),
                                           np.array(self.z_path_yaw_axis),
                                           color="green", linewidth=0.5)

        self.x_resultant = []
        self.y_resultant = []
        self.z_resultant = []
        self.path_resultant, = self.ax.plot(np.array(self.x_resultant),
                                            np.array(self.y_resultant),
                                            np.array(self.z_resultant),
                                            color="black", linewidth=1)

        self.cube_p1 = np.array([1., 1., 1.])
        self.cube_p2 = np.array([-1., 1., 1.])
        self.cube_p3 = np.array([-1., -1., 1.])
        self.cube_p4 = np.array([1., -1., 1.])
        self.cube_p5 = np.array([1., 1., -1.])
        self.cube_p6 = np.array([-1., 1., -1.])
        self.cube_p7 = np.array([-1., -1., -1.])
        self.cube_p8 = np.array([1., -1., -1.])

        self.line_1to2, = self.ax.plot(np.array([self.cube_p1[0], self.cube_p2[0]]),
                                       np.array([self.cube_p1[1], self.cube_p2[1]]),
                                       np.array([self.cube_p1[2], self.cube_p2[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_2to3, = self.ax.plot(np.array([self.cube_p2[0], self.cube_p3[0]]),
                                       np.array([self.cube_p2[1], self.cube_p3[1]]),
                                       np.array([self.cube_p2[2], self.cube_p3[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_3to4, = self.ax.plot(np.array([self.cube_p3[0], self.cube_p4[0]]),
                                       np.array([self.cube_p3[1], self.cube_p4[1]]),
                                       np.array([self.cube_p3[2], self.cube_p4[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_4to1, = self.ax.plot(np.array([self.cube_p4[0], self.cube_p1[0]]),
                                       np.array([self.cube_p4[1], self.cube_p1[1]]),
                                       np.array([self.cube_p4[2], self.cube_p1[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_5to6, = self.ax.plot(np.array([self.cube_p5[0], self.cube_p6[0]]),
                                       np.array([self.cube_p5[1], self.cube_p6[1]]),
                                       np.array([self.cube_p5[2], self.cube_p6[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_6to7, = self.ax.plot(np.array([self.cube_p6[0], self.cube_p7[0]]),
                                       np.array([self.cube_p6[1], self.cube_p7[1]]),
                                       np.array([self.cube_p6[2], self.cube_p7[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_7to8, = self.ax.plot(np.array([self.cube_p7[0], self.cube_p8[0]]),
                                       np.array([self.cube_p7[1], self.cube_p8[1]]),
                                       np.array([self.cube_p7[2], self.cube_p8[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_8to5, = self.ax.plot(np.array([self.cube_p8[0], self.cube_p5[0]]),
                                       np.array([self.cube_p8[1], self.cube_p5[1]]),
                                       np.array([self.cube_p8[2], self.cube_p5[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_1to5, = self.ax.plot(np.array([self.cube_p1[0], self.cube_p5[0]]),
                                       np.array([self.cube_p1[1], self.cube_p5[1]]),
                                       np.array([self.cube_p1[2], self.cube_p5[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_2to6, = self.ax.plot(np.array([self.cube_p2[0], self.cube_p6[0]]),
                                       np.array([self.cube_p2[1], self.cube_p6[1]]),
                                       np.array([self.cube_p2[2], self.cube_p6[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_3to7, = self.ax.plot(np.array([self.cube_p3[0], self.cube_p7[0]]),
                                       np.array([self.cube_p3[1], self.cube_p7[1]]),
                                       np.array([self.cube_p3[2], self.cube_p7[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")
        self.line_4to8, = self.ax.plot(np.array([self.cube_p4[0], self.cube_p8[0]]),
                                       np.array([self.cube_p4[1], self.cube_p8[1]]),
                                       np.array([self.cube_p4[2], self.cube_p8[2]]),
                                       color="darkorange", linewidth=1, linestyle="--")

    def roll(self, angle):
        self.roll_axis = self.roll_axis / np.linalg.norm(self.roll_axis)
        rot_matrix = Rotation.from_rotvec(angle * self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self.update_quiver()
        self._update_path()

        self.cube_p1 = rot_matrix.apply(self.cube_p1)
        self.cube_p2 = rot_matrix.apply(self.cube_p2)
        self.cube_p3 = rot_matrix.apply(self.cube_p3)
        self.cube_p4 = rot_matrix.apply(self.cube_p4)
        self.cube_p5 = rot_matrix.apply(self.cube_p5)
        self.cube_p6 = rot_matrix.apply(self.cube_p6)
        self.cube_p7 = rot_matrix.apply(self.cube_p7)
        self.cube_p8 = rot_matrix.apply(self.cube_p8)
        self.update_cube()

    def pitch(self, angle):
        self.pitch_axis = self.pitch_axis / np.linalg.norm(self.pitch_axis)
        rot_matrix = Rotation.from_rotvec(angle * self.pitch_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self.update_quiver()
        self._update_path()

        self.cube_p1 = rot_matrix.apply(self.cube_p1)
        self.cube_p2 = rot_matrix.apply(self.cube_p2)
        self.cube_p3 = rot_matrix.apply(self.cube_p3)
        self.cube_p4 = rot_matrix.apply(self.cube_p4)
        self.cube_p5 = rot_matrix.apply(self.cube_p5)
        self.cube_p6 = rot_matrix.apply(self.cube_p6)
        self.cube_p7 = rot_matrix.apply(self.cube_p7)
        self.cube_p8 = rot_matrix.apply(self.cube_p8)
        self.update_cube()

    def yaw(self, angle):
        self.yaw_axis = self.yaw_axis / np.linalg.norm(self.yaw_axis)
        rot_matrix = Rotation.from_rotvec(angle * self.yaw_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)

        self.update_quiver()
        self._update_path()

        self.cube_p1 = rot_matrix.apply(self.cube_p1)
        self.cube_p2 = rot_matrix.apply(self.cube_p2)
        self.cube_p3 = rot_matrix.apply(self.cube_p3)
        self.cube_p4 = rot_matrix.apply(self.cube_p4)
        self.cube_p5 = rot_matrix.apply(self.cube_p5)
        self.cube_p6 = rot_matrix.apply(self.cube_p6)
        self.cube_p7 = rot_matrix.apply(self.cube_p7)
        self.cube_p8 = rot_matrix.apply(self.cube_p8)
        self.update_cube()

    def rot_x(self, angle):
        rot_matrix = Rotation.from_rotvec(angle * vector_x_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self.update_quiver()
        self._update_path()

        self.cube_p1 = rot_matrix.apply(self.cube_p1)
        self.cube_p2 = rot_matrix.apply(self.cube_p2)
        self.cube_p3 = rot_matrix.apply(self.cube_p3)
        self.cube_p4 = rot_matrix.apply(self.cube_p4)
        self.cube_p5 = rot_matrix.apply(self.cube_p5)
        self.cube_p6 = rot_matrix.apply(self.cube_p6)
        self.cube_p7 = rot_matrix.apply(self.cube_p7)
        self.cube_p8 = rot_matrix.apply(self.cube_p8)
        self.update_cube()

    def rot_y(self, angle):
        rot_matrix = Rotation.from_rotvec(angle * vector_y_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self.update_quiver()
        self._update_path()

        self.cube_p1 = rot_matrix.apply(self.cube_p1)
        self.cube_p2 = rot_matrix.apply(self.cube_p2)
        self.cube_p3 = rot_matrix.apply(self.cube_p3)
        self.cube_p4 = rot_matrix.apply(self.cube_p4)
        self.cube_p5 = rot_matrix.apply(self.cube_p5)
        self.cube_p6 = rot_matrix.apply(self.cube_p6)
        self.cube_p7 = rot_matrix.apply(self.cube_p7)
        self.cube_p8 = rot_matrix.apply(self.cube_p8)
        self.update_cube()

    def rot_z(self, angle):
        rot_matrix = Rotation.from_rotvec(angle * vector_z_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self.update_quiver()
        self._update_path()

        self.cube_p1 = rot_matrix.apply(self.cube_p1)
        self.cube_p2 = rot_matrix.apply(self.cube_p2)
        self.cube_p3 = rot_matrix.apply(self.cube_p3)
        self.cube_p4 = rot_matrix.apply(self.cube_p4)
        self.cube_p5 = rot_matrix.apply(self.cube_p5)
        self.cube_p6 = rot_matrix.apply(self.cube_p6)
        self.cube_p7 = rot_matrix.apply(self.cube_p7)
        self.cube_p8 = rot_matrix.apply(self.cube_p8)
        self.update_cube()

    def reset(self):
        self.xyz = np.array([0., 0., 0.])
        self.roll_axis = np.array([1., 0., 0.])
        self.pitch_axis = np.array([0., 1., 0.])
        self.yaw_axis = np.array([0., 0., 1.])

        self.x_path_roll_axis = []
        self.y_path_roll_axis = []
        self.z_path_roll_axis = []

        self.x_path_pitch_axis = []
        self.y_path_pitch_axis = []
        self.z_path_pitch_axis = []

        self.x_path_yaw_axis = []
        self.y_path_yaw_axis = []
        self.z_path_yaw_axis = []

        self.x_resultant = []
        self.y_resultant = []
        self.z_resultant = []

        self.update_quiver()
        self._update_path()

        self.cube_p1 = np.array([1., 1., 1.])
        self.cube_p2 = np.array([-1., 1., 1.])
        self.cube_p3 = np.array([-1., -1., 1.])
        self.cube_p4 = np.array([1., -1., 1.])
        self.cube_p5 = np.array([1., 1., -1.])
        self.cube_p6 = np.array([-1., 1., -1.])
        self.cube_p7 = np.array([-1., -1., -1.])
        self.cube_p8 = np.array([1., -1., -1.])
        self.update_cube()

    def update_quiver(self):
        self.qvr_roll_axis.remove()
        self.qvr_roll_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                            self.roll_axis[0], self.roll_axis[1], self.roll_axis[2],
                                            length=1, color="red", normalize=True, linewidth=2)
        self.qvr_pitch_axis.remove()
        self.qvr_pitch_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                             self.pitch_axis[0], self.pitch_axis[1], self.pitch_axis[2],
                                             length=1, color="blue", normalize=True, linewidth=2)
        self.qvr_yaw_axis.remove()
        self.qvr_yaw_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.yaw_axis[0], self.yaw_axis[1], self.yaw_axis[2],
                                           length=1, color="green", normalize=True, linewidth=2)

        if self.is_rot_roll_pitch_yaw:
            self.vector_a = self.roll_axis * rot_velocity_a * wave_a
            self.vector_b = self.pitch_axis * rot_velocity_b * wave_b
            self.vector_c = self.yaw_axis * rot_velocity_c * wave_c
        else:
            self.vector_a = vector_x_axis * rot_velocity_a * wave_a
            self.vector_b = vector_y_axis * rot_velocity_b * wave_b
            self.vector_c = vector_z_axis * rot_velocity_c * wave_c

        self.qvr_a_axis.remove()
        self.qvr_a_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.vector_a[0], self.vector_a[1], self.vector_a[2],
                                         length=1, color="red", linestyle="-.", linewidth=1)
        self.qvr_b_axis.remove()
        self.qvr_b_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.vector_b[0], self.vector_b[1], self.vector_b[2],
                                         length=1, color="blue", linestyle="-.", linewidth=1)
        self.qvr_c_axis.remove()
        self.qvr_c_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.vector_c[0], self.vector_c[1], self.vector_c[2],
                                         length=1, color="green", linestyle="-.", linewidth=1)

        self.vector_resultant = self.vector_a + self.vector_b + self.vector_c
        self.qvr_resultant_vector.remove()
        self.qvr_resultant_vector = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                                   self.vector_resultant[0], self.vector_resultant[1],
                                                   self.vector_resultant[2],
                                                   length=1, color="black", normalize=False, linewidth=1,
                                                   linestyle="-.")

    def _update_path(self):
        if self.is_path_a_on:
            self.x_path_roll_axis.append(self.roll_axis[0])
            self.y_path_roll_axis.append(self.roll_axis[1])
            self.z_path_roll_axis.append(self.roll_axis[2])
        self.path_roll_axis.set_xdata(np.array(self.x_path_roll_axis))
        self.path_roll_axis.set_ydata(np.array(self.y_path_roll_axis))
        self.path_roll_axis.set_3d_properties(np.array(self.z_path_roll_axis))

        if self.is_path_b_on:
            self.x_path_pitch_axis.append(self.pitch_axis[0])
            self.y_path_pitch_axis.append(self.pitch_axis[1])
            self.z_path_pitch_axis.append(self.pitch_axis[2])
        self.path_pitch_axis.set_xdata(np.array(self.x_path_pitch_axis))
        self.path_pitch_axis.set_ydata(np.array(self.y_path_pitch_axis))
        self.path_pitch_axis.set_3d_properties(np.array(self.z_path_pitch_axis))

        if self.is_path_c_on:
            self.x_path_yaw_axis.append(self.yaw_axis[0])
            self.y_path_yaw_axis.append(self.yaw_axis[1])
            self.z_path_yaw_axis.append(self.yaw_axis[2])
        self.path_yaw_axis.set_xdata(np.array(self.x_path_yaw_axis))
        self.path_yaw_axis.set_ydata(np.array(self.y_path_yaw_axis))
        self.path_yaw_axis.set_3d_properties(np.array(self.z_path_yaw_axis))

        if self.is_path_r_on:
            self.x_resultant.append(self.vector_resultant[0])
            self.y_resultant.append(self.vector_resultant[1])
            self.z_resultant.append(self.vector_resultant[2])
        self.path_resultant.set_xdata(np.array(self.x_resultant))
        self.path_resultant.set_ydata(np.array(self.y_resultant))
        self.path_resultant.set_3d_properties(np.array(self.z_resultant))

    def clear_path(self):
        self.x_path_roll_axis = []
        self.y_path_roll_axis = []
        self.z_path_roll_axis = []

        self.x_path_pitch_axis = []
        self.y_path_pitch_axis = []
        self.z_path_pitch_axis = []

        self.x_path_yaw_axis = []
        self.y_path_yaw_axis = []
        self.z_path_yaw_axis = []

        self.x_resultant = []
        self.y_resultant = []
        self.z_resultant = []

        self._update_path()

    def update_cube(self):
        self.line_1to2.set_xdata(np.array([self.cube_p1[0], self.cube_p2[0]]))
        self.line_1to2.set_ydata(np.array([self.cube_p1[1], self.cube_p2[1]]))
        self.line_1to2.set_3d_properties(np.array([self.cube_p1[2], self.cube_p2[2]]))

        self.line_2to3.set_xdata(np.array([self.cube_p2[0], self.cube_p3[0]]))
        self.line_2to3.set_ydata(np.array([self.cube_p2[1], self.cube_p3[1]]))
        self.line_2to3.set_3d_properties(np.array([self.cube_p2[2], self.cube_p3[2]]))

        self.line_3to4.set_xdata(np.array([self.cube_p3[0], self.cube_p4[0]]))
        self.line_3to4.set_ydata(np.array([self.cube_p3[1], self.cube_p4[1]]))
        self.line_3to4.set_3d_properties(np.array([self.cube_p3[2], self.cube_p4[2]]))

        self.line_4to1.set_xdata(np.array([self.cube_p4[0], self.cube_p1[0]]))
        self.line_4to1.set_ydata(np.array([self.cube_p4[1], self.cube_p1[1]]))
        self.line_4to1.set_3d_properties(np.array([self.cube_p4[2], self.cube_p1[2]]))

        self.line_5to6.set_xdata(np.array([self.cube_p5[0], self.cube_p6[0]]))
        self.line_5to6.set_ydata(np.array([self.cube_p5[1], self.cube_p6[1]]))
        self.line_5to6.set_3d_properties(np.array([self.cube_p5[2], self.cube_p6[2]]))

        self.line_6to7.set_xdata(np.array([self.cube_p6[0], self.cube_p7[0]]))
        self.line_6to7.set_ydata(np.array([self.cube_p6[1], self.cube_p7[1]]))
        self.line_6to7.set_3d_properties(np.array([self.cube_p6[2], self.cube_p7[2]]))

        self.line_7to8.set_xdata(np.array([self.cube_p7[0], self.cube_p8[0]]))
        self.line_7to8.set_ydata(np.array([self.cube_p7[1], self.cube_p8[1]]))
        self.line_7to8.set_3d_properties(np.array([self.cube_p7[2], self.cube_p8[2]]))

        self.line_8to5.set_xdata(np.array([self.cube_p8[0], self.cube_p5[0]]))
        self.line_8to5.set_ydata(np.array([self.cube_p8[1], self.cube_p5[1]]))
        self.line_8to5.set_3d_properties(np.array([self.cube_p8[2], self.cube_p5[2]]))

        self.line_1to5.set_xdata(np.array([self.cube_p1[0], self.cube_p5[0]]))
        self.line_1to5.set_ydata(np.array([self.cube_p1[1], self.cube_p5[1]]))
        self.line_1to5.set_3d_properties(np.array([self.cube_p1[2], self.cube_p5[2]]))

        self.line_2to6.set_xdata(np.array([self.cube_p2[0], self.cube_p6[0]]))
        self.line_2to6.set_ydata(np.array([self.cube_p2[1], self.cube_p6[1]]))
        self.line_2to6.set_3d_properties(np.array([self.cube_p2[2], self.cube_p6[2]]))

        self.line_3to7.set_xdata(np.array([self.cube_p3[0], self.cube_p7[0]]))
        self.line_3to7.set_ydata(np.array([self.cube_p3[1], self.cube_p7[1]]))
        self.line_3to7.set_3d_properties(np.array([self.cube_p3[2], self.cube_p7[2]]))

        self.line_4to8.set_xdata(np.array([self.cube_p4[0], self.cube_p8[0]]))
        self.line_4to8.set_ydata(np.array([self.cube_p4[1], self.cube_p8[1]]))
        self.line_4to8.set_3d_properties(np.array([self.cube_p4[2], self.cube_p8[2]]))

    def set_is_rot_roll_pitch_yaw(self, value):
        self.is_rot_roll_pitch_yaw = value

    def set_is_path_a_on(self, value):
        self.is_path_a_on = value

    def set_is_path_b_on(self, value):
        self.is_path_b_on = value

    def set_is_path_c_on(self, value):
        self.is_path_c_on = value

    def set_is_path_r_on(self, value):
        self.is_path_r_on = value


class WaveMonitor:
    def __init__(self, ax=None):
        self.ax = ax

        self.x_wave = np.arange(0., 360, 1.)

        self.y_wave_a = rot_velocity_a * np.cos(wave_number_a * np.deg2rad(self.x_wave + wave_phase_a))
        self.y_wave_b = rot_velocity_b * np.cos(wave_number_b * np.deg2rad(self.x_wave + wave_phase_b))
        self.y_wave_c = rot_velocity_c * np.cos(wave_number_c * np.deg2rad(self.x_wave + wave_phase_c))
        self.y_wave_r = np.sqrt(self.y_wave_a ** 2 + self.y_wave_b ** 2 + self.y_wave_c ** 2)

        self.plt_wave_a, = self.ax.plot(self.x_wave, self.y_wave_a, color="red", linestyle="-.", linewidth=1)
        self.plt_wave_b, = self.ax.plot(self.x_wave, self.y_wave_b, color="blue", linestyle="-.", linewidth=1)
        self.plt_wave_c, = self.ax.plot(self.x_wave, self.y_wave_c, color="green", linestyle="-.", linewidth=1)
        self.plt_wave_r, = self.ax.plot(self.x_wave, self.y_wave_r, color="black", linestyle="-.", linewidth=1)

    def update(self):
        self.y_wave_a = rot_velocity_a * np.cos(wave_number_a * np.deg2rad(self.x_wave + wave_phase_a))
        self.y_wave_b = rot_velocity_b * np.cos(wave_number_b * np.deg2rad(self.x_wave + wave_phase_b))
        self.y_wave_c = rot_velocity_c * np.cos(wave_number_c * np.deg2rad(self.x_wave + wave_phase_c))
        self.y_wave_r = np.sqrt(self.y_wave_a ** 2 + self.y_wave_b ** 2 + self.y_wave_c ** 2)

        self.plt_wave_a.set_data(self.x_wave, self.y_wave_a)
        self.plt_wave_b.set_data(self.x_wave, self.y_wave_b)
        self.plt_wave_c.set_data(self.x_wave, self.y_wave_c)
        self.plt_wave_r.set_data(self.x_wave, self.y_wave_r)


def create_center_lines():
    line_axis_x = art3d.Line3D([0., 0.], [0., 0.], [z_min, z_max], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_x)
    line_axis_y = art3d.Line3D([x_min, x_max], [0., 0.], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_y)
    line_axis_z = art3d.Line3D([0., 0.], [y_min, y_max], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_z)


def create_animation_control():
    frm_anim = ttk.Labelframe(root, relief="ridge", text="Animation", labelanchor="n")
    frm_anim.pack(side="left", fill=tk.Y)
    btn_play = tk.Button(frm_anim, text="Play/Pause", command=switch)
    btn_play.pack(fill=tk.X)
    btn_reset = tk.Button(frm_anim, text="Reset", command=reset)
    btn_reset.pack(fill=tk.X)
    btn_clear = tk.Button(frm_anim, text="Clear path", command=lambda: three_arrow.clear_path())
    btn_clear.pack(fill=tk.X)


def set_angle_step(angle):
    global angle_step_deg
    angle_step_deg = angle


def set_v_roll(velocity):
    global rot_velocity_a
    rot_velocity_a = velocity
    three_arrow.update_quiver()
    three_arrow.update_cube()


def set_v_pitch(velocity):
    global rot_velocity_b
    rot_velocity_b = velocity
    three_arrow.update_quiver()
    three_arrow.update_cube()


def set_v_yaw(velocity):
    global rot_velocity_c
    rot_velocity_c = velocity
    three_arrow.update_quiver()
    three_arrow.update_cube()


def set_roll_cw_initial():
    three_arrow.roll(np.deg2rad(angle_step_deg))
    three_arrow.clear_path()
    three_arrow.update_cube()


def set_roll_ccw_initial():
    three_arrow.roll(np.deg2rad(- angle_step_deg))
    three_arrow.clear_path()
    three_arrow.update_cube()


def set_pitch_up_initial():
    three_arrow.pitch(np.deg2rad(- angle_step_deg))
    three_arrow.clear_path()
    three_arrow.update_cube()


def set_pitch_down_initial():
    three_arrow.pitch(np.deg2rad(angle_step_deg))
    three_arrow.clear_path()
    three_arrow.update_cube()


def set_yaw_right_initial():
    three_arrow.yaw(np.deg2rad(- angle_step_deg))
    three_arrow.clear_path()
    three_arrow.update_cube()


def set_yaw_left_initial():
    three_arrow.yaw(np.deg2rad(angle_step_deg))
    three_arrow.clear_path()
    three_arrow.update_cube()


def switch_rot_axis():
    global is_rot_roll_pitch_yaw
    is_rot_roll_pitch_yaw = not is_rot_roll_pitch_yaw
    three_arrow.set_is_rot_roll_pitch_yaw(is_rot_roll_pitch_yaw)
    three_arrow.update_quiver()
    three_arrow.update_cube()


def set_wave_number_a(value):
    global wave_number_a
    wave_number_a = value
    three_arrow.update_quiver()
    three_arrow.update_cube()
    wave_monitor.update()


def set_wave_number_b(value):
    global wave_number_b
    wave_number_b = value
    three_arrow.update_quiver()
    three_arrow.update_cube()
    wave_monitor.update()


def set_wave_number_c(value):
    global wave_number_c
    wave_number_c = value
    three_arrow.update_quiver()
    three_arrow.update_cube()
    wave_monitor.update()


def set_wave_phase_a(value):
    global wave_phase_a
    wave_phase_a = value
    three_arrow.update_quiver()
    three_arrow.update_cube()
    wave_monitor.update()


def set_wave_phase_b(value):
    global wave_phase_b
    wave_phase_b = value
    three_arrow.update_quiver()
    three_arrow.update_cube()
    wave_monitor.update()


def set_wave_phase_c(value):
    global wave_phase_c
    wave_phase_c = value
    three_arrow.update_quiver()
    three_arrow.update_cube()
    wave_monitor.update()


def set_is_wave(value):
    global is_wave
    is_wave = value
    three_arrow.update_quiver()
    three_arrow.update_cube()
    wave_monitor.update()


def set_is_path_a(value):
    global is_path_a_on
    is_path_a_on = value
    three_arrow.set_is_path_a_on(is_path_a_on)


def set_is_path_b(value):
    global is_path_b_on
    is_path_b_on = value
    three_arrow.set_is_path_b_on(is_path_b_on)


def set_is_path_c(value):
    global is_path_c_on
    is_path_c_on = value
    three_arrow.set_is_path_c_on(is_path_c_on)


def set_is_path_r(value):
    global is_path_r_on
    is_path_r_on = value
    three_arrow.set_is_path_r_on(is_path_r_on)


def create_parameter_setter():
    global var_axis_op, var_turn_op, var_apply_v

    frm_angle = ttk.Labelframe(root, relief="ridge", text="Angle per step", labelanchor='n')
    frm_angle.pack(side='left', fill=tk.Y)

    var_angle_stp = tk.StringVar(root)
    var_angle_stp.set(str(angle_step_deg))
    spn_stp_angle = tk.Spinbox(
        frm_angle, textvariable=var_angle_stp, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_angle_step(float(var_angle_stp.get())), width=5
    )
    spn_stp_angle.pack(side="left")

    frm_dir = ttk.Labelframe(root, relief="ridge", text="Initial direction", labelanchor='n')
    frm_dir.pack(side='left', fill=tk.Y)

    frm_roll = ttk.Labelframe(frm_dir, relief="ridge", text="Roll", labelanchor='n')
    frm_roll.pack(side='left', fill=tk.Y)
    btn_roll_cw = tk.Button(frm_roll, text="CW", command=set_roll_cw_initial)
    btn_roll_cw.pack(fill=tk.X)
    btn_roll_ccw = tk.Button(frm_roll, text="CCW", command=set_roll_ccw_initial)
    btn_roll_ccw.pack(fill=tk.X)

    frm_pitch = ttk.Labelframe(frm_dir, relief="ridge", text="Pitch", labelanchor='n')
    frm_pitch.pack(side='left', fill=tk.Y)
    btn_roll_pitch_up = tk.Button(frm_pitch, text="Up", command=set_pitch_up_initial)
    btn_roll_pitch_up.pack(fill=tk.X)
    btn_roll_pitch_down = tk.Button(frm_pitch, text="Down", command=set_pitch_down_initial)
    btn_roll_pitch_down.pack(fill=tk.X)

    frm_yaw = ttk.Labelframe(frm_dir, relief="ridge", text="Yaw", labelanchor='n')
    frm_yaw.pack(side='left', fill=tk.Y)
    btn_roll_yaw_right = tk.Button(frm_yaw, text="Right", command=set_yaw_right_initial)
    btn_roll_yaw_right.pack(fill=tk.X)
    btn_roll_yaw_left = tk.Button(frm_yaw, text="Left", command=set_yaw_left_initial)
    btn_roll_yaw_left.pack(fill=tk.X)

    frm_axis = ttk.Labelframe(root, relief="ridge", text="Rotation axis", labelanchor='n')
    frm_axis.pack(side='left', fill=tk.Y)

    # var_axis_op = tk.IntVar()
    rd_op_axis_rpy = tk.Radiobutton(frm_axis, text="Roll,Pitch,Yaw", value=1, variable=var_axis_op,
                                    command=lambda: three_arrow.set_is_rot_roll_pitch_yaw(True))
    rd_op_axis_rpy.pack(anchor=tk.W)

    rd_op_axis_xyz = tk.Radiobutton(frm_axis, text="x,y,z", value=2, variable=var_axis_op,
                                    command=lambda: three_arrow.set_is_rot_roll_pitch_yaw(False))
    rd_op_axis_xyz.pack(anchor=tk.W)

    var_axis_op.set(1)

    frm_v = ttk.Labelframe(root, relief="ridge", text="Rotation velocity", labelanchor='n')
    frm_v.pack(side='left', fill=tk.Y)

    lbl_vr = tk.Label(frm_v, text="A(Roll,x)")
    lbl_vr.pack(side="left")

    var_vr = tk.StringVar(root)
    var_vr.set(str(rot_velocity_a))
    spn_vr = tk.Spinbox(
        frm_v, textvariable=var_vr, format="%.0f", from_=-100, to=100, increment=1,
        command=lambda: set_v_roll(float(var_vr.get())), width=5
    )
    spn_vr.pack(side="left")

    lbl_vp = tk.Label(frm_v, text="B(Pitch,y)")
    lbl_vp.pack(side="left")

    var_vp = tk.StringVar(root)
    var_vp.set(str(rot_velocity_b))
    spn_vp = tk.Spinbox(
        frm_v, textvariable=var_vp, format="%.0f", from_=-100, to=100, increment=1,
        command=lambda: set_v_pitch(float(var_vp.get())), width=5
    )
    spn_vp.pack(side="left")

    lbl_vy = tk.Label(frm_v, text="C(Yaw,z)")
    lbl_vy.pack(side="left")

    var_vy = tk.StringVar(root)
    var_vy.set(str(rot_velocity_c))
    spn_vy = tk.Spinbox(
        frm_v, textvariable=var_vy, format="%.0f", from_=-100, to=100, increment=1,
        command=lambda: set_v_yaw(float(var_vy.get())), width=5
    )
    spn_vy.pack(side="left")

    frm_wave = ttk.Labelframe(root, relief="ridge", text="Wave (Wave number, Phase(deg))", labelanchor='n')
    frm_wave.pack(side='left', fill=tk.Y)

    var_chk_wave = tk.BooleanVar(root)
    chk_wave = tk.Checkbutton(frm_wave, text="Apply", variable=var_chk_wave,
                              command=lambda: set_is_wave(var_chk_wave.get()))
    chk_wave.pack(side='left')
    var_chk_wave.set(is_wave)

    frm_wave_a = ttk.Labelframe(frm_wave, relief="ridge", text="A", labelanchor='n')
    frm_wave_a.pack(side='left', fill=tk.Y)
    var_wn_a = tk.StringVar(root)
    var_wn_a.set(str(wave_number_a))
    spn_wn_a = tk.Spinbox(
        frm_wave_a, textvariable=var_wn_a, format="%.1f", from_=-10, to=10, increment=.1,
        command=lambda: set_wave_number_a(float(var_wn_a.get())), width=5
    )
    spn_wn_a.pack()

    var_wp_a = tk.StringVar(root)
    var_wp_a.set(str(wave_phase_a))
    spn_wp_a = tk.Spinbox(
        frm_wave_a, textvariable=var_wp_a, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_wave_phase_a(float(var_wp_a.get())), width=5
    )
    spn_wp_a.pack()

    frm_wave_b = ttk.Labelframe(frm_wave, relief="ridge", text="B", labelanchor='n')
    frm_wave_b.pack(side='left', fill=tk.Y)
    var_wn_b = tk.StringVar(root)
    var_wn_b.set(str(wave_number_b))
    spn_wn_b = tk.Spinbox(
        frm_wave_b, textvariable=var_wn_b, format="%.1f", from_=-10, to=10, increment=.1,
        command=lambda: set_wave_number_b(float(var_wn_b.get())), width=5
    )
    spn_wn_b.pack()

    var_wp_b = tk.StringVar(root)
    var_wp_b.set(str(wave_phase_b))
    spn_wp_b = tk.Spinbox(
        frm_wave_b, textvariable=var_wp_b, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_wave_phase_b(float(var_wp_b.get())), width=5
    )
    spn_wp_b.pack()

    frm_wave_c = ttk.Labelframe(frm_wave, relief="ridge", text="C", labelanchor='n')
    frm_wave_c.pack(side='left', fill=tk.Y)
    var_wn_c = tk.StringVar(root)
    var_wn_c.set(str(wave_number_c))
    spn_wn_c = tk.Spinbox(
        frm_wave_c, textvariable=var_wn_c, format="%.1f", from_=-10, to=10, increment=.1,
        command=lambda: set_wave_number_c(float(var_wn_c.get())), width=5
    )
    spn_wn_c.pack()

    var_wp_c = tk.StringVar(root)
    var_wp_c.set(str(wave_phase_c))
    spn_wp_c = tk.Spinbox(
        frm_wave_c, textvariable=var_wp_c, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_wave_phase_c(float(var_wp_c.get())), width=5
    )
    spn_wp_c.pack()

    frm_turn = ttk.Labelframe(root, relief="ridge", text="Turn of rotation", labelanchor='n')
    frm_turn.pack(side='left', fill=tk.Y)

    # var_turn_op = tk.IntVar()
    rd_op_rpy = tk.Radiobutton(frm_turn, text="A->B->C", value=1, variable=var_turn_op)
    rd_op_rpy.pack(anchor=tk.W)

    rd_op_pyr = tk.Radiobutton(frm_turn, text="C->B->A", value=2, variable=var_turn_op)
    rd_op_pyr.pack(anchor=tk.W)

    var_turn_op.set(1)

    frm_path = ttk.Labelframe(root, relief="ridge", text="Path of arrows", labelanchor='n')
    frm_path.pack(side='left', fill=tk.Y)

    var_chk_path_a = tk.BooleanVar(root)
    chk_path_a = tk.Checkbutton(frm_path, text="A", variable=var_chk_path_a,
                                command=lambda: set_is_path_a(var_chk_path_a.get()))
    chk_path_a.pack(side='left')
    var_chk_path_a.set(is_path_a_on)

    var_chk_path_b = tk.BooleanVar(root)
    chk_path_b = tk.Checkbutton(frm_path, text="B", variable=var_chk_path_b,
                                command=lambda: set_is_path_b(var_chk_path_b.get()))
    chk_path_b.pack(side='left')
    var_chk_path_b.set(is_path_b_on)

    var_chk_path_c = tk.BooleanVar(root)
    chk_path_c = tk.Checkbutton(frm_path, text="C", variable=var_chk_path_c,
                                command=lambda: set_is_path_c(var_chk_path_c.get()))
    chk_path_c.pack(side='left')
    var_chk_path_c.set(is_path_c_on)

    var_chk_path_r = tk.BooleanVar(root)
    chk_path_r = tk.Checkbutton(frm_path, text="Resultant", variable=var_chk_path_r,
                                command=lambda: set_is_path_r(var_chk_path_r.get()))
    chk_path_r.pack(side='left')
    var_chk_path_r.set(is_path_r_on)


def create_circle(ax, x, y, z, z_dir, edge_col, fill_flag, line_width, line_style, label):
    if label != "":
        c_spin_axis_guide = Circle((x, y), 1., ec=edge_col, fill=fill_flag,
                                   linewidth=line_width, linestyle=line_style, label=label)
    else:
        c_spin_axis_guide = Circle((x, y), 1., ec=edge_col, fill=fill_flag,
                                   linewidth=line_width, linestyle=line_style)
    ax.add_patch(c_spin_axis_guide)
    art3d.pathpatch_2d_to_3d(c_spin_axis_guide, z=z, zdir=z_dir)


def draw_static_diagrams():
    create_center_lines()
    create_circle(ax0, 0., 0., 0., "x", "gray", False, 0.5,
                  "--", "")
    create_circle(ax0, 0., 0., 0., "y", "gray", False, 0.5,
                  "--", "")
    create_circle(ax0, 0., 0., 0., "z", "gray", False, 0.5,
                  "--", "")


def update_diagrams():
    global angle_acc_deg, wave_a, wave_b, wave_c
    # angle = np.deg2rad(cnt.get()) % (2. * np.pi)
    angle = np.deg2rad(angle_step_deg)
    if not is_wave:
        wave_a, wave_b, wave_c = 1., 1., 1.
    else:
        angle_acc_deg = (angle_acc_deg + angle_step_deg)
        # print(angle_acc_deg % 360)
        phase = np.deg2rad(angle_acc_deg) * adjustment
        wave_a = np.cos(wave_number_a * phase + np.deg2rad(wave_phase_a))
        wave_b = np.cos(wave_number_b * phase + np.deg2rad(wave_phase_b))
        wave_c = np.cos(wave_number_c * phase + np.deg2rad(wave_phase_c))
    if var_axis_op.get() == 1:
        if var_turn_op.get() == 1:
            # Roll->Pitch->Yaw
            # print("Roll->Pitch->Yaw")
            three_arrow.roll(rot_velocity_a * angle * wave_a)
            three_arrow.pitch(rot_velocity_b * angle * wave_b)
            three_arrow.yaw(rot_velocity_c * angle * wave_c)
        else:
            # Yaw->Pitch->Roll
            # print("Yaw->Pitch->Roll")
            three_arrow.yaw(rot_velocity_c * angle * wave_a)
            three_arrow.pitch(rot_velocity_b * angle * wave_b)
            three_arrow.roll(rot_velocity_a * angle * wave_c)
    else:
        if var_turn_op.get() == 1:
            # x->y->z
            # print("x->y->z")
            three_arrow.rot_x(rot_velocity_a * angle * wave_a)
            three_arrow.rot_y(rot_velocity_b * angle * wave_b)
            three_arrow.rot_z(rot_velocity_c * angle * wave_c)
        else:
            # z->y->x
            # print("z->y->x")
            three_arrow.rot_z(rot_velocity_c * angle * wave_a)
            three_arrow.rot_y(rot_velocity_b * angle * wave_b)
            three_arrow.rot_x(rot_velocity_a * angle * wave_c)


def reset():
    global is_play, angle_acc_deg
    cnt.reset()
    three_arrow.reset()
    angle_acc_deg = 0.


def switch():
    global is_play
    is_play = not is_play


def update(f):
    if is_play:
        cnt.count_up()
        update_diagrams()


""" main loop """
if __name__ == "__main__":
    cnt = Counter(ax=ax0, is3d=True, xy=np.array([x_min, y_max]), z=z_max, label="Step=")
    draw_static_diagrams()
    create_animation_control()
    create_parameter_setter()

    three_arrow = ThreeArrow(ax0, np.array([0., 0., 0.]), 0.)
    wave_monitor = WaveMonitor(ax1)

    dummy1, = ax0.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
                       color="red", linewidth=1, linestyle="-.", label="Rotation vector A")
    dummy2, = ax0.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
                       color="blue", linewidth=1, linestyle="-.", label="Rotation vector B")
    dummy3, = ax0.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
                       color="green", linewidth=1, linestyle="-.", label="Rotation vector C")
    dummy0, = ax0.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
                       color="black", linewidth=1, linestyle="-.", label="Resultant rotation vector")

    ax0.legend(loc='lower right', fontsize=8)

    anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
    root.mainloop()

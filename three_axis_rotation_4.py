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
phase_deg = 0.
phase_deg_step = 1.

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
ax1.set_title("Oscillation of rotation velocity")
ax1.set_xlabel("Phase(deg)")
ax1.set_ylabel("Velocity")
ax1.set_xlim(0, 360,)
ax1.set_ylim(-4., 4.)
# ax1.set_aspect("equal")
ax1.set_aspect(30)
ax1.grid()
ax1.set_xticks(np.arange(0, 360, 45))

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
    def __init__(self, ax=None, xyz=None):
        self.ax = ax
        self.xyz = xyz
        self.is_rotate_xyz = True

        self.adjustment = np.pi

        self.axis_a = np.array([1., 0., 0.])
        self.axis_b = np.array([0., 1., 0.])
        self.axis_c = np.array([0., 0., 1.])

        self.qvr_axis_a = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.axis_a[0], self.axis_a[1], self.axis_a[2],
                                         length=1, color="red", normalize=True, linewidth=2, alpha=0.5)

        self.qvr_axis_b = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.axis_b[0], self.axis_b[1], self.axis_b[2],
                                         length=1, color="blue", normalize=True, linewidth=2, alpha=0.5)

        self.qvr_axis_c = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.axis_c[0], self.axis_c[1], self.axis_c[2],
                                         length=1, color="green", normalize=True, linewidth=2, alpha=0.5)

        self.rotation_velocity_a = 1.
        self.rotation_velocity_b = 1.
        self.rotation_velocity_c = 1.

        if self.is_rotate_xyz:
            self.vector_a = vector_x_axis * self.rotation_velocity_a
            self.vector_b = vector_y_axis * self.rotation_velocity_b
            self.vector_c = vector_z_axis * self.rotation_velocity_c
        else:
            self.vector_a = self.axis_a * self.rotation_velocity_a
            self.vector_b = self.axis_b * self.rotation_velocity_b
            self.vector_c = self.axis_c * self.rotation_velocity_c

        self.qvr_vector_a = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.vector_a[0], self.vector_a[1], self.vector_a[2],
                                           length=1, color="red", linestyle="-.", linewidth=1, normalize=False)

        self.qvr_vector_b = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.vector_b[0], self.vector_b[1], self.vector_b[2],
                                           length=1, color="blue", linestyle="-.", linewidth=1, normalize=False)

        self.qvr_vector_c = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.vector_c[0], self.vector_c[1], self.vector_c[2],
                                           length=1, color="green", linestyle="-.", linewidth=1, normalize=False)

        self.vector_resultant = self.vector_a + self.vector_b + self.vector_c

        self.qvr_vector_resultant = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                                   self.vector_resultant[0], self.vector_resultant[1],
                                                   self.vector_resultant[2], length=1, color="black",
                                                   normalize=False, linewidth=1, linestyle="-.")

        self.is_path_a_on = True
        self.is_path_b_on = True
        self.is_path_c_on = True
        self.is_path_r_on = True

        self.x_axis_a = []
        self.y_axis_a = []
        self.z_axis_a = []
        self.path_axis_a, = self.ax.plot(np.array(self.x_axis_a), np.array(self.y_axis_a), np.array(self.z_axis_a),
                                         color="red", linewidth=0.5, alpha=1)

        self.x_axis_b = []
        self.y_axis_b = []
        self.z_axis_b = []
        self.path_axis_b, = self.ax.plot(np.array(self.x_axis_b), np.array(self.y_axis_b), np.array(self.z_axis_b),
                                         color="blue", linewidth=0.5, alpha=1)

        self.x_axis_c = []
        self.y_axis_c = []
        self.z_axis_c = []
        self.path_axis_c, = self.ax.plot(np.array(self.x_axis_c), np.array(self.y_axis_c), np.array(self.z_axis_c),
                                         color="green", linewidth=0.5, alpha=1)

        self.x_resultant = []
        self.y_resultant = []
        self.z_resultant = []
        self.path_resultant, = self.ax.plot(np.array(self.x_resultant), np.array(self.y_resultant), np.array(self.z_resultant),
                                            color="black", linewidth=1)

    def update_quiver(self):
        self.qvr_axis_a.remove()
        self.qvr_axis_a = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.axis_a[0], self.axis_a[1], self.axis_a[2],
                                         length=1, color="red", normalize=True, linewidth=2, alpha=0.5)

        self.qvr_axis_b.remove()
        self.qvr_axis_b = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.axis_b[0], self.axis_b[1], self.axis_b[2],
                                         length=1, color="blue", normalize=True, linewidth=2, alpha=0.5)

        self.qvr_axis_c.remove()
        self.qvr_axis_c = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                         self.axis_c[0], self.axis_c[1], self.axis_c[2],
                                         length=1, color="green", normalize=True, linewidth=2, alpha=0.5)

        if self.is_rotate_xyz:
            self.vector_a = vector_x_axis * self.rotation_velocity_a
            self.vector_b = vector_y_axis * self.rotation_velocity_b
            self.vector_c = vector_z_axis * self.rotation_velocity_c
        else:
            self.vector_a = self.axis_a * self.rotation_velocity_a
            self.vector_b = self.axis_b * self.rotation_velocity_b
            self.vector_c = self.axis_c * self.rotation_velocity_c

        self.qvr_vector_a.remove()
        self.qvr_vector_a = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.vector_a[0], self.vector_a[1], self.vector_a[2],
                                           length=1, color="red", linestyle="-.", linewidth=1, normalize=False)

        self.qvr_vector_b.remove()
        self.qvr_vector_b = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.vector_b[0], self.vector_b[1], self.vector_b[2],
                                           length=1, color="blue", linestyle="-.", linewidth=1, normalize=False)

        self.qvr_vector_c.remove()
        self.qvr_vector_c = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.vector_c[0], self.vector_c[1], self.vector_c[2],
                                           length=1, color="green", linestyle="-.", linewidth=1, normalize=False)

        self.vector_resultant = self.vector_a + self.vector_b + self.vector_c

        self.qvr_vector_resultant.remove()
        self.qvr_vector_resultant = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                                   self.vector_resultant[0], self.vector_resultant[1],
                                                   self.vector_resultant[2], length=1, color="black",
                                                   normalize=False, linewidth=1, linestyle="-.")

    def set_velocity_a(self, value):
        self.rotation_velocity_a = value

    def set_velocity_b(self, value):
        self.rotation_velocity_b = value

    def set_velocity_c(self, value):
        self.rotation_velocity_c = value

    def rotate_a_cw(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_a)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()

    def rotate_a_ccw(self):
        rot_matrix = Rotation.from_rotvec(- np.deg2rad(phase_deg_step) * self.vector_a)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()

    def rotate_b_cw(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_b)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()

    def rotate_b_ccw(self):
        rot_matrix = Rotation.from_rotvec(- np.deg2rad(phase_deg_step) * self.vector_b)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()

    def rotate_c_cw(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_c)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()

    def rotate_c_ccw(self):
        rot_matrix = Rotation.from_rotvec(- np.deg2rad(phase_deg_step) * self.vector_c)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()

    def rotate_resultant_axis(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_resultant * self.adjustment)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()
        self.update_path()

    def rotate_abc(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_a * self.adjustment)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_b * self.adjustment)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_c * self.adjustment)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()
        self.update_path()

    def rotate_cba(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_c * self.adjustment)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_b * self.adjustment)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        rot_matrix = Rotation.from_rotvec(np.deg2rad(phase_deg_step) * self.vector_a * self.adjustment)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_quiver()
        self.update_path()

    def set_is_rotate_xyz(self, value):
        self.is_rotate_xyz = value

    def reset(self):
        self.xyz = np.array([0., 0., 0.])
        self.axis_a = np.array([1., 0., 0.])
        self.axis_b = np.array([0., 1., 0.])
        self.axis_c = np.array([0., 0., 1.])

        self.update_quiver()

    def update_path(self):
        if self.is_path_a_on:
            self.x_axis_a.append(self.axis_a[0])
            self.y_axis_a.append(self.axis_a[1])
            self.z_axis_a.append(self.axis_a[2])
        self.path_axis_a.set_xdata(np.array(self.x_axis_a))
        self.path_axis_a.set_ydata(np.array(self.y_axis_a))
        self.path_axis_a.set_3d_properties(np.array(self.z_axis_a))

        if self.is_path_b_on:
            self.x_axis_b.append(self.axis_b[0])
            self.y_axis_b.append(self.axis_b[1])
            self.z_axis_b.append(self.axis_b[2])
        self.path_axis_b.set_xdata(np.array(self.x_axis_b))
        self.path_axis_b.set_ydata(np.array(self.y_axis_b))
        self.path_axis_b.set_3d_properties(np.array(self.z_axis_b))

        if self.is_path_c_on:
            self.x_axis_c.append(self.axis_c[0])
            self.y_axis_c.append(self.axis_c[1])
            self.z_axis_c.append(self.axis_c[2])
        self.path_axis_c.set_xdata(np.array(self.x_axis_c))
        self.path_axis_c.set_ydata(np.array(self.y_axis_c))
        self.path_axis_c.set_3d_properties(np.array(self.z_axis_c))

        if self.is_path_r_on:
            self.x_resultant.append(self.vector_resultant[0])
            self.y_resultant.append(self.vector_resultant[1])
            self.z_resultant.append(self.vector_resultant[2])
        self.path_resultant.set_xdata(np.array(self.x_resultant))
        self.path_resultant.set_ydata(np.array(self.y_resultant))
        self.path_resultant.set_3d_properties(np.array(self.z_resultant))

    def clear_path(self):
        self.x_axis_a = []
        self.y_axis_a = []
        self.z_axis_a = []

        self.x_axis_b = []
        self.y_axis_b = []
        self.z_axis_b = []

        self.x_axis_c = []
        self.y_axis_c = []
        self.z_axis_c = []

        self.x_resultant = []
        self.y_resultant = []
        self.z_resultant = []

        self.update_path()

    def set_is_path_a(self, value):
        self.is_path_a_on = value

    def set_is_path_b(self, value):
        self.is_path_b_on = value

    def set_is_path_c(self, value):
        self.is_path_c_on = value

    def set_is_path_r(self, value):
        self.is_path_r_on = value


class RotationVelocityController:
    def __init__(self, ax=None):
        self.ax = ax

        self.amplitude_a = 1.
        self.amplitude_b = 1.
        self.amplitude_c = 1.

        self.wave_number_a = 0.
        self.wave_number_b = 0.
        self.wave_number_c = 0.

        self.phase_deg_a = 0.
        self.phase_deg_b = 0.
        self.phase_deg_c = 0.

        self.phase_deg_current = 0.

        self.x_wave = np.arange(0., 360, 1.)
        self.y_wave_a = self.amplitude_a * np.cos(self.wave_number_a * np.deg2rad(self.x_wave) +
                                                  np.deg2rad(self.phase_deg_a))
        self.y_wave_b = self.amplitude_b * np.cos(self.wave_number_b * np.deg2rad(self.x_wave) +
                                                  np.deg2rad(self.phase_deg_b))
        self.y_wave_c = self.amplitude_c * np.cos(self.wave_number_c * np.deg2rad(self.x_wave) +
                                                  np.deg2rad(self.phase_deg_c))
        self.y_wave_r = np.sqrt(self.y_wave_a ** 2 + self.y_wave_b ** 2 + self.y_wave_c ** 2)

        self.plt_wave_a, = self.ax.plot(self.x_wave, self.y_wave_a, color="red", linestyle="-.", linewidth=1,
                                        label="Rotation vector A")
        self.plt_wave_b, = self.ax.plot(self.x_wave, self.y_wave_b, color="blue", linestyle="-.", linewidth=1,
                                        label="Rotation vector B")
        self.plt_wave_c, = self.ax.plot(self.x_wave, self.y_wave_c, color="green", linestyle="-.", linewidth=1,
                                        label="Rotation vector C")
        self.plt_wave_r, = self.ax.plot(self.x_wave, self.y_wave_r, color="black", linestyle="-.", linewidth=1,
                                        label="Sqrt(A**2+B**2+C**2)")

        self.plt_phase_current, = self.ax.plot(np.array([self.phase_deg_current, self.phase_deg_current]),
                                               np.array([-4., 4.]), color="magenta", linestyle="-",
                                               linewidth=1, label="Current phase")

    def update_wave(self):
        self.y_wave_a = self.amplitude_a * np.cos(self.wave_number_a * np.deg2rad(self.x_wave) +
                                                  np.deg2rad(self.phase_deg_a))
        self.y_wave_b = self.amplitude_b * np.cos(self.wave_number_b * np.deg2rad(self.x_wave) +
                                                  np.deg2rad(self.phase_deg_b))
        self.y_wave_c = self.amplitude_c * np.cos(self.wave_number_c * np.deg2rad(self.x_wave) +
                                                  np.deg2rad(self.phase_deg_c))
        self.y_wave_r = np.sqrt(self.y_wave_a ** 2 + self.y_wave_b ** 2 + self.y_wave_c ** 2)

        self.plt_wave_a.set_data(self.x_wave, self.y_wave_a)
        self.plt_wave_b.set_data(self.x_wave, self.y_wave_b)
        self.plt_wave_c.set_data(self.x_wave, self.y_wave_c)
        self.plt_wave_r.set_data(self.x_wave, self.y_wave_r)

    def update_phase_current(self):
        self.plt_phase_current.set_data(np.array([self.phase_deg_current, self.phase_deg_current]),
                                        np.array([-4., 4.]))

    def set_amplitude_a(self, value):
        self.amplitude_a = value
        self.update_wave()
        three_arrow.set_velocity_a(rotation_velocity_controller.get_velocity_a())
        three_arrow.update_quiver()

    def set_amplitude_b(self, value):
        self.amplitude_b = value
        self.update_wave()
        three_arrow.set_velocity_b(rotation_velocity_controller.get_velocity_b())
        three_arrow.update_quiver()

    def set_amplitude_c(self, value):
        self.amplitude_c = value
        self.update_wave()
        three_arrow.set_velocity_c(rotation_velocity_controller.get_velocity_c())
        three_arrow.update_quiver()

    def set_wave_number_a(self, value):
        self.wave_number_a = value
        self.update_wave()
        three_arrow.set_velocity_a(rotation_velocity_controller.get_velocity_a())
        three_arrow.update_quiver()

    def set_wave_number_b(self, value):
        self.wave_number_b = value
        self.update_wave()
        three_arrow.set_velocity_b(rotation_velocity_controller.get_velocity_b())
        three_arrow.update_quiver()

    def set_wave_number_c(self, value):
        self.wave_number_c = value
        self.update_wave()
        three_arrow.set_velocity_c(rotation_velocity_controller.get_velocity_c())
        three_arrow.update_quiver()

    def set_phase_deg_a(self, value):
        self.phase_deg_a = value
        self.update_wave()
        three_arrow.set_velocity_a(rotation_velocity_controller.get_velocity_a())
        three_arrow.update_quiver()

    def set_phase_deg_b(self, value):
        self.phase_deg_b = value
        self.update_wave()
        three_arrow.set_velocity_b(rotation_velocity_controller.get_velocity_b())
        three_arrow.update_quiver()

    def set_phase_deg_c(self, value):
        self.phase_deg_c = value
        self.update_wave()
        three_arrow.set_velocity_c(rotation_velocity_controller.get_velocity_c())
        three_arrow.update_quiver()

    def set_phase_deg_current(self, value):
        self.phase_deg_current = value % 360
        self.update_phase_current()
        three_arrow.set_velocity_a(rotation_velocity_controller.get_velocity_a())
        three_arrow.set_velocity_b(rotation_velocity_controller.get_velocity_b())
        three_arrow.set_velocity_c(rotation_velocity_controller.get_velocity_c())
        three_arrow.update_quiver()

    def get_phase_current(self):
        return self.phase_deg_current

    def get_velocity_a(self):
        velocity = self.amplitude_a * np.cos(self.wave_number_a * np.deg2rad(self.phase_deg_current) +
                                             np.deg2rad(self.phase_deg_a))
        return velocity

    def get_velocity_b(self):
        velocity = self.amplitude_b * np.cos(self.wave_number_b * np.deg2rad(self.phase_deg_current) +
                                             np.deg2rad(self.phase_deg_b))
        return velocity

    def get_velocity_c(self):
        velocity = self.amplitude_c * np.cos(self.wave_number_c * np.deg2rad(self.phase_deg_current) +
                                             np.deg2rad(self.phase_deg_c))
        return velocity

    def reset(self):
        self.phase_deg_current = 0.
        self.update_phase_current()
        three_arrow.set_velocity_a(rotation_velocity_controller.get_velocity_a())
        three_arrow.set_velocity_b(rotation_velocity_controller.get_velocity_b())
        three_arrow.set_velocity_c(rotation_velocity_controller.get_velocity_c())
        three_arrow.update_quiver()


def set_phase_deg_step(value):
    global phase_deg_step
    phase_deg_step = value


def create_parameter_setter():
    # Rotation axis
    frm_axis = ttk.Labelframe(root, relief="ridge", text="Rotation vector(A,B,C) axis", labelanchor='n')
    frm_axis.pack(side='left', fill=tk.Y)

    # var_axis_op = tk.IntVar()
    rd_op_axis_xyz = tk.Radiobutton(frm_axis, text="x,y,z", value=1, variable=var_axis_op,
                                    command=lambda: three_arrow.set_is_rotate_xyz(True))
    rd_op_axis_xyz.pack(anchor=tk.W)
    rd_op_axis_rpy = tk.Radiobutton(frm_axis, text="Arrows(Red,Blue,Green)", value=2, variable=var_axis_op,
                                    command=lambda: three_arrow.set_is_rotate_xyz(False))
    rd_op_axis_rpy.pack(anchor=tk.W)
    var_axis_op.set(1)

    # Turn of rotation
    frm_turn = ttk.Labelframe(root, relief="ridge", text="Turn of rotation", labelanchor='n')
    frm_turn.pack(side='left', fill=tk.Y)

    # var_turn_op = tk.IntVar()
    rd_op_rpy = tk.Radiobutton(frm_turn, text="Resultant vector", value=1, variable=var_turn_op)
    rd_op_rpy.pack(anchor=tk.W)

    rd_op_rpy = tk.Radiobutton(frm_turn, text="Rotation vector A->B->C", value=2, variable=var_turn_op)
    rd_op_rpy.pack(anchor=tk.W)

    rd_op_pyr = tk.Radiobutton(frm_turn, text="Rotation vector C->B->A", value=3, variable=var_turn_op)
    rd_op_pyr.pack(anchor=tk.W)

    var_turn_op.set(1)

    # Phase per step
    frm_phase_step = ttk.Labelframe(root, relief="ridge", text="Phase per step(deg)", labelanchor='n')
    frm_phase_step.pack(side='left', fill=tk.Y)

    var_phase_stp = tk.StringVar(root)
    var_phase_stp.set(str(phase_deg_step))
    spn_stp_angle = tk.Spinbox(
        frm_phase_step, textvariable=var_phase_stp, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_phase_deg_step(float(var_phase_stp.get())), width=5
    )
    spn_stp_angle.pack(side="left")

    # Initial direction
    frm_dir = ttk.Labelframe(root, relief="ridge", text="Initial direction", labelanchor='n')
    frm_dir.pack(side='left', fill=tk.Y)

    frm_roll = ttk.Labelframe(frm_dir, relief="ridge", text="A", labelanchor='n')
    frm_roll.pack(side='left', fill=tk.Y)
    btn_roll_cw = tk.Button(frm_roll, text="CW", command=lambda: three_arrow.rotate_a_cw())
    btn_roll_cw.pack(fill=tk.X)
    btn_roll_ccw = tk.Button(frm_roll, text="CCW", command=lambda: three_arrow.rotate_a_ccw())
    btn_roll_ccw.pack(fill=tk.X)

    frm_pitch = ttk.Labelframe(frm_dir, relief="ridge", text="B", labelanchor='n')
    frm_pitch.pack(side='left', fill=tk.Y)
    btn_roll_pitch_up = tk.Button(frm_pitch, text="CW", command=lambda: three_arrow.rotate_b_cw())
    btn_roll_pitch_up.pack(fill=tk.X)
    btn_roll_pitch_down = tk.Button(frm_pitch, text="CCW", command=lambda: three_arrow.rotate_b_ccw())
    btn_roll_pitch_down.pack(fill=tk.X)

    frm_yaw = ttk.Labelframe(frm_dir, relief="ridge", text="C", labelanchor='n')
    frm_yaw.pack(side='left', fill=tk.Y)
    btn_roll_yaw_right = tk.Button(frm_yaw, text="CW", command=lambda: three_arrow.rotate_c_cw())
    btn_roll_yaw_right.pack(fill=tk.X)
    btn_roll_yaw_left = tk.Button(frm_yaw, text="CCW", command=lambda: three_arrow.rotate_c_ccw())
    btn_roll_yaw_left.pack(fill=tk.X)

    # Wave parameter
    frm_wave = ttk.Labelframe(root, relief="ridge", text="Wave (Amp., Wave number, Phase(deg))", labelanchor='n')
    frm_wave.pack(side='left', fill=tk.Y)

    frm_wave_a = ttk.Labelframe(frm_wave, relief="ridge", text="A", labelanchor='n')
    frm_wave_a.pack(side='left', fill=tk.Y)

    var_amp_a = tk.StringVar(root)
    var_amp_a.set(str(1.))
    spn_amp_a = tk.Spinbox(
        frm_wave_a, textvariable=var_amp_a, format="%.1f", from_=-10, to=10, increment=1,
        command=lambda: rotation_velocity_controller.set_amplitude_a(float(var_amp_a.get())), width=5
    )
    spn_amp_a.pack()

    var_wn_a = tk.StringVar(root)
    var_wn_a.set(str(0))
    spn_wn_a = tk.Spinbox(
        frm_wave_a, textvariable=var_wn_a, format="%.1f", from_=-10, to=10, increment=1,
        command=lambda: rotation_velocity_controller.set_wave_number_a(float(var_wn_a.get())), width=5
    )
    spn_wn_a.pack()

    var_wp_a = tk.StringVar(root)
    var_wp_a.set(str(0))
    spn_wp_a = tk.Spinbox(
        frm_wave_a, textvariable=var_wp_a, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: rotation_velocity_controller.set_phase_deg_a(float(var_wp_a.get())), width=5
    )
    spn_wp_a.pack()

    frm_wave_b = ttk.Labelframe(frm_wave, relief="ridge", text="B", labelanchor='n')
    frm_wave_b.pack(side='left', fill=tk.Y)

    var_amp_b = tk.StringVar(root)
    var_amp_b.set(str(1))
    spn_amp_b = tk.Spinbox(
        frm_wave_b, textvariable=var_amp_b, format="%.1f", from_=-10, to=10, increment=1,
        command=lambda: rotation_velocity_controller.set_amplitude_b(float(var_amp_b.get())), width=5
    )
    spn_amp_b.pack()

    var_wn_b = tk.StringVar(root)
    var_wn_b.set(str(0))
    spn_wn_b = tk.Spinbox(
        frm_wave_b, textvariable=var_wn_b, format="%.1f", from_=-10, to=10, increment=1,
        command=lambda: rotation_velocity_controller.set_wave_number_b(float(var_wn_b.get())), width=5
    )
    spn_wn_b.pack()

    var_wp_b = tk.StringVar(root)
    var_wp_b.set(str(0))
    spn_wp_b = tk.Spinbox(
        frm_wave_b, textvariable=var_wp_b, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: rotation_velocity_controller.set_phase_deg_b(float(var_wp_b.get())), width=5
    )
    spn_wp_b.pack()

    frm_wave_c = ttk.Labelframe(frm_wave, relief="ridge", text="C", labelanchor='n')
    frm_wave_c.pack(side='left', fill=tk.Y)

    var_amp_c = tk.StringVar(root)
    var_amp_c.set(str(1))
    spn_amp_c = tk.Spinbox(
        frm_wave_c, textvariable=var_amp_c, format="%.1f", from_=-10, to=10, increment=1,
        command=lambda: rotation_velocity_controller.set_amplitude_c(float(var_amp_c.get())), width=5
    )
    spn_amp_c.pack()

    var_wn_c = tk.StringVar(root)
    var_wn_c.set(str(0))
    spn_wn_c = tk.Spinbox(
        frm_wave_c, textvariable=var_wn_c, format="%.1f", from_=-10, to=10, increment=1,
        command=lambda: rotation_velocity_controller.set_wave_number_c(float(var_wn_c.get())), width=5
    )
    spn_wn_c.pack()

    var_wp_c = tk.StringVar(root)
    var_wp_c.set(str(0))
    spn_wp_c = tk.Spinbox(
        frm_wave_c, textvariable=var_wp_c, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: rotation_velocity_controller.set_phase_deg_c(float(var_wp_c.get())), width=5
    )
    spn_wp_c.pack()

    # Path of arrows
    frm_path = ttk.Labelframe(root, relief="ridge", text="Path of arrows", labelanchor='n')
    frm_path.pack(side='left', fill=tk.Y)

    var_chk_path_a = tk.BooleanVar(root)
    chk_path_a = tk.Checkbutton(frm_path, text="A", variable=var_chk_path_a,
                                command=lambda: three_arrow.set_is_path_a(var_chk_path_a.get()))
    chk_path_a.pack(anchor=tk.W)
    var_chk_path_a.set(True)

    var_chk_path_b = tk.BooleanVar(root)
    chk_path_b = tk.Checkbutton(frm_path, text="B", variable=var_chk_path_b,
                                command=lambda: three_arrow.set_is_path_b(var_chk_path_b.get()))
    chk_path_b.pack(anchor=tk.W)
    var_chk_path_b.set(True)

    var_chk_path_c = tk.BooleanVar(root)
    chk_path_c = tk.Checkbutton(frm_path, text="C", variable=var_chk_path_c,
                                command=lambda: three_arrow.set_is_path_c(var_chk_path_c.get()))
    chk_path_c.pack(anchor=tk.W)
    var_chk_path_c.set(True)

    var_chk_path_r = tk.BooleanVar(root)
    chk_path_r = tk.Checkbutton(frm_path, text="Resultant", variable=var_chk_path_r,
                                command=lambda: three_arrow.set_is_path_r(var_chk_path_r.get()))
    chk_path_r.pack(anchor=tk.W)
    var_chk_path_r.set(True)


def create_animation_control():
    frm_anim = ttk.Labelframe(root, relief="ridge", text="Animation", labelanchor="n")
    frm_anim.pack(side="left", fill=tk.Y)
    btn_play = tk.Button(frm_anim, text="Play/Pause", command=switch)
    btn_play.pack(fill=tk.X)
    btn_reset = tk.Button(frm_anim, text="Reset", command=reset)
    btn_reset.pack(fill=tk.X)
    btn_clear = tk.Button(frm_anim, text="Clear path", command=lambda: three_arrow.clear_path())
    btn_clear.pack(fill=tk.X)


def create_center_lines():
    line_axis_x = art3d.Line3D([0., 0.], [0., 0.], [z_min, z_max], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_x)
    line_axis_y = art3d.Line3D([x_min, x_max], [0., 0.], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_y)
    line_axis_z = art3d.Line3D([0., 0.], [y_min, y_max], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(line_axis_z)


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
    global phase_deg
    phase_deg += phase_deg_step
    rotation_velocity_controller.set_phase_deg_current(phase_deg)
    # print(rotation_velocity_controller.get_phase_current())
    three_arrow.set_velocity_a(rotation_velocity_controller.get_velocity_a())
    three_arrow.set_velocity_b(rotation_velocity_controller.get_velocity_b())
    three_arrow.set_velocity_c(rotation_velocity_controller.get_velocity_c())
    three_arrow.update_quiver()
    if var_turn_op.get() == 1:
        three_arrow.rotate_resultant_axis()
    elif var_turn_op.get() == 2:
        three_arrow.rotate_abc()
    else:
        three_arrow.rotate_cba()


def reset():
    global is_play, phase_deg
    cnt.reset()
    phase_deg = 0.
    rotation_velocity_controller.reset()
    three_arrow.reset()


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

    rotation_velocity_controller = RotationVelocityController(ax1)
    three_arrow = ThreeArrow(ax0, np.array([0., 0., 0.]))

    dummy1, = ax0.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
                       color="red", linewidth=1, linestyle="-.", label="Rotation vector A")
    dummy2, = ax0.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
                       color="blue", linewidth=1, linestyle="-.", label="Rotation vector B")
    dummy3, = ax0.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
                       color="green", linewidth=1, linestyle="-.", label="Rotation vector C")
    dummy0, = ax0.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
                       color="black", linewidth=1, linestyle="-.", label="Resultant rotation vector")

    ax0.legend(loc='lower right', fontsize=8)
    ax1.legend(loc='lower right', fontsize=8)

    anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
    root.mainloop()

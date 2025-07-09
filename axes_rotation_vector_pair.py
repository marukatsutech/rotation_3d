""" Axes rotation (vector pair) """
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
is_path_a = False
is_path_b = False
is_path_c = False

""" Axis vectors """
vector_x_axis = np.array([1., 0., 0.])
vector_y_axis = np.array([0., 1., 0.])
vector_z_axis = np.array([0., 0., 1.])

""" Other parameters """
phase_step_deg = 2.

""" Create figure and axes """
title_ax0 = "Axes rotation (vector pair)"
title_tk = title_ax0

x_min = -2.
x_max = 2.
y_min = -2.
y_max = 2.
z_min = -2.
z_max = 2.

fig = Figure()
ax0 = fig.add_subplot(111, projection='3d')
ax0.set_box_aspect((1, 1, 1))
ax0.grid()
ax0.set_title(title_ax0)
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")
ax0.set_xlim(x_min, x_max)
ax0.set_ylim(y_min, y_max)
ax0.set_zlim(z_min, z_max)

""" Embed in Tkinter """
root = tk.Tk()
root.title(title_tk)
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(expand=True, fill="both")

toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()

""" Global objects of Tkinter """
var_rot_a = tk.BooleanVar(root)
var_rot_b = tk.BooleanVar(root)
var_rot_c = tk.BooleanVar(root)

var_path_a = tk.BooleanVar(root)
var_path_b = tk.BooleanVar(root)
var_path_c = tk.BooleanVar(root)

var_phase_step = tk.StringVar(root)

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


class ThreeAxes:
    def __init__(self, ax=None):
        self.ax = ax

        self.origin = np.array([0., 0., 0.])
        self.phase_step_deg = 2

        self.scale_a = 1
        self.scale_b = 1
        self.scale_c = 1

        self.scale_ab = 1
        self.scale_bc = 1
        self.scale_ca = 1

        self.axis_a = np.array([1., 0., 0.])
        self.axis_b = np.array([0., 1., 0.])
        self.axis_c = np.array([0., 0., 1.])

        self.resultant_ab = self.axis_a + self.axis_b
        self.resultant_bc = self.axis_b + self.axis_c
        self.resultant_ca = self.axis_c + self.axis_a

        self.x_axis_a = [self.axis_a[0], - self.axis_a[0]]
        self.y_axis_a = [self.axis_a[1], - self.axis_a[1]]
        self.z_axis_a = [self.axis_a[2], - self.axis_a[2]]
        self.line_axis_a, = self.ax.plot(self.x_axis_a, self.y_axis_a, self.z_axis_a,
                                         label='Axis A', color="red", ls="-.", linewidth=2)

        self.x_axis_b = [self.axis_b[0], - self.axis_b[0]]
        self.y_axis_b = [self.axis_b[1], - self.axis_b[1]]
        self.z_axis_b = [self.axis_b[2], - self.axis_b[2]]
        self.line_axis_b, = self.ax.plot(self.x_axis_b, self.y_axis_b, self.z_axis_b,
                                         label='Axis B', color="green", ls="-.", linewidth=2)

        self.x_axis_c = [self.axis_c[0], - self.axis_c[0]]
        self.y_axis_c = [self.axis_c[1], - self.axis_c[1]]
        self.z_axis_c = [self.axis_c[2], - self.axis_c[2]]
        self.line_axis_c, = self.ax.plot(self.x_axis_c, self.y_axis_c, self.z_axis_c,
                                         label='Axis C', color="blue", ls="-.", linewidth=2)

        self.vector_a = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                       self.axis_a[0] * self.scale_a, self.axis_a[1] * self.scale_a,
                                       self.axis_a[2] * self.scale_a,
                                       length=1, color="red", normalize=True, linewidth=3, alpha=1.0)

        self.vector_b = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                       self.axis_b[0] * self.scale_b, self.axis_b[1] * self.scale_b,
                                       self.axis_b[2] * self.scale_b,
                                       length=1, color="green", normalize=True, linewidth=3, alpha=1.0)

        self.vector_c = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                       self.axis_c[0] * self.scale_c, self.axis_c[1] * self.scale_c,
                                       self.axis_c[2] * self.scale_c,
                                       length=1, color="blue", normalize=True, linewidth=3, alpha=1.0)

        self.vector_ab = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                        self.resultant_ab[0] * self.scale_ab, self.resultant_ab[1] * self.scale_ab,
                                        self.resultant_ab[2] * self.scale_ab,
                                        length=1, color="darkorange", normalize=False, linewidth=1, alpha=1.0)

        self.vector_bc = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                        self.resultant_bc[0] * self.scale_bc, self.resultant_bc[1] * self.scale_bc,
                                        self.resultant_bc[2] * self.scale_bc,
                                        length=1, color="cyan", normalize=False, linewidth=1, alpha=1.0)

        self.vector_ca = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                        self.resultant_ca[0] * self.scale_ca, self.resultant_ca[1] * self.scale_ca,
                                        self.resultant_ca[2] * self.scale_ca,
                                        length=1, color="magenta", normalize=False, linewidth=1, alpha=1.0)

    def set_phase_step_deg(self, value):
        self.phase_step_deg = value

    def update_vectors(self):
        self.vector_a.remove()
        self.vector_b.remove()
        self.vector_c.remove()

        self.vector_a = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                       self.axis_a[0] * self.scale_a, self.axis_a[1] * self.scale_a,
                                       self.axis_a[2] * self.scale_a,
                                       length=1, color="red", normalize=True, linewidth=3, alpha=1.0)

        self.vector_b = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                       self.axis_b[0] * self.scale_b, self.axis_b[1] * self.scale_b,
                                       self.axis_b[2] * self.scale_b,
                                       length=1, color="green", normalize=True, linewidth=3, alpha=1.0)

        self.vector_c = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                       self.axis_c[0] * self.scale_c, self.axis_c[1] * self.scale_c,
                                       self.axis_c[2] * self.scale_c,
                                       length=1, color="blue", normalize=True, linewidth=3, alpha=1.0)

        self.vector_ab.remove()
        self.vector_bc.remove()
        self.vector_ca.remove()

        self.vector_ab = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                        self.resultant_ab[0] * self.scale_ab, self.resultant_ab[1] * self.scale_ab,
                                        self.resultant_ab[2] * self.scale_ab,
                                        length=1, color="darkorange", normalize=False, linewidth=1, alpha=1.0)

        self.vector_bc = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                        self.resultant_bc[0] * self.scale_bc, self.resultant_bc[1] * self.scale_bc,
                                        self.resultant_bc[2] * self.scale_bc,
                                        length=1, color="cyan", normalize=False, linewidth=1, alpha=1.0)

        self.vector_ca = self.ax.quiver(self.origin[0], self.origin[1], self.origin[2],
                                        self.resultant_ca[0] * self.scale_ca, self.resultant_ca[1] * self.scale_ca,
                                        self.resultant_ca[2] * self.scale_ca,
                                        length=1, color="magenta", normalize=False, linewidth=1, alpha=1.0)

    def rotate_all_axis_a(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(self.phase_step_deg) * self.axis_a)
        # self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_axes()
        self.update_vectors()

    def rotate_all_axis_b(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(self.phase_step_deg) * self.axis_b)
        self.axis_a = rot_matrix.apply(self.axis_a)
        # self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_axes()
        self.update_vectors()

    def rotate_all_axis_c(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(self.phase_step_deg) * self.axis_c)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        # self.axis_c = rot_matrix.apply(self.axis_c)

        self.update_axes()
        self.update_vectors()

    def rotate_all_resultant_ab(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(self.phase_step_deg) * self.resultant_ab)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.resultant_ab = self.axis_a + self.axis_b
        self.resultant_bc = self.axis_b + self.axis_c
        self.resultant_ca = self.axis_c + self.axis_a

        self.update_axes()
        self.update_vectors()

    def rotate_all_resultant_bc(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(self.phase_step_deg) * self.resultant_bc)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.resultant_ab = self.axis_a + self.axis_b
        self.resultant_bc = self.axis_b + self.axis_c
        self.resultant_ca = self.axis_c + self.axis_a

        self.update_axes()
        self.update_vectors()

    def rotate_all_resultant_ca(self):
        rot_matrix = Rotation.from_rotvec(np.deg2rad(self.phase_step_deg) * self.resultant_ca)
        self.axis_a = rot_matrix.apply(self.axis_a)
        self.axis_b = rot_matrix.apply(self.axis_b)
        self.axis_c = rot_matrix.apply(self.axis_c)

        self.resultant_ab = self.axis_a + self.axis_b
        self.resultant_bc = self.axis_b + self.axis_c
        self.resultant_ca = self.axis_c + self.axis_a

        self.update_axes()
        self.update_vectors()

    def update_axes(self):
        self.x_axis_a = [self.axis_a[0], - self.axis_a[0]]
        self.y_axis_a = [self.axis_a[1], - self.axis_a[1]]
        self.z_axis_a = [self.axis_a[2], - self.axis_a[2]]
        self.line_axis_a.set_data_3d(self.x_axis_a, self.y_axis_a, self.z_axis_a)

        self.x_axis_b = [self.axis_b[0], - self.axis_b[0]]
        self.y_axis_b = [self.axis_b[1], - self.axis_b[1]]
        self.z_axis_b = [self.axis_b[2], - self.axis_b[2]]
        self.line_axis_b.set_data_3d(self.x_axis_b, self.y_axis_b, self.z_axis_b)

        self.x_axis_c = [self.axis_c[0], - self.axis_c[0]]
        self.y_axis_c = [self.axis_c[1], - self.axis_c[1]]
        self.z_axis_c = [self.axis_c[2], - self.axis_c[2]]
        self.line_axis_c.set_data_3d(self.x_axis_c, self.y_axis_c, self.z_axis_c)

    def reset(self):
        self.origin = np.array([0., 0., 0.])
        self.axis_a = np.array([1., 0., 0.])
        self.axis_b = np.array([0., 1., 0.])
        self.axis_c = np.array([0., 0., 1.])

        self.update_axes()
        self.update_vectors()

    def set_scale_a(self, value):
        self.scale_a = value
        self.update_vectors()

    def set_scale_b(self, value):
        self.scale_b = value
        self.update_vectors()

    def set_scale_c(self, value):
        self.scale_c = value
        self.update_vectors()

    def get_axis_a(self):
        return self.axis_a

    def get_axis_b(self):
        return self.axis_b

    def get_axis_c(self):
        return self.axis_c

    def get_resultant_ab(self):
        return self.resultant_ab

    def get_resultant_bc(self):
        return self.resultant_bc

    def get_resultant_ca(self):
        return self.resultant_ca


class Path:
    def __init__(self, ax, color):
        self.ax = ax
        self.color = color

        self.is_draw_path = False

        self.x_path = []
        self.y_path = []
        self.z_path = []
        self.path, = self.ax.plot(np.array(self.x_path), np.array(self.y_path), np.array(self.z_path),
                                  color=self.color, linewidth=1)

    def append_path(self, position):
        if self.is_draw_path:
            self.x_path.append(position[0])
            self.y_path.append(position[1])
            self.z_path.append(position[2])
            self.update_path()

    def update_path(self):
        self.path.set_data_3d(np.array(self.x_path), np.array(self.y_path), np.array(self.z_path))

    def clear_path(self):
        self.x_path = []
        self.y_path = []
        self.z_path = []
        self.update_path()

    def set_is_draw_path(self, value):
        self.is_draw_path = value


def set_phase_step_deg(value):
    global phase_step_deg
    phase_step_deg = value
    three_axes.set_phase_step_deg(phase_step_deg)


def clear_path():
    path_a.clear_path()
    path_b.clear_path()
    path_c.clear_path()


def create_parameter_setter():
    global var_rot_a, var_rot_b, var_rot_c

    # Paths
    frm_path = ttk.Labelframe(root, relief="ridge", text="Path", labelanchor='n')
    frm_path.pack(side="left", fill=tk.Y)

    # var_path_a = tk.BooleanVar(root)
    chk_path_a = tk.Checkbutton(frm_path, text="Axis A", variable=var_path_a)
    chk_path_a.pack(anchor=tk.W)
    var_path_a.set(False)

    # var_path_b = tk.BooleanVar(root)
    chk_path_b = tk.Checkbutton(frm_path, text="Axis B", variable=var_path_b)
    chk_path_b.pack(anchor=tk.W)
    var_path_b.set(False)

    # var_path_c = tk.BooleanVar(root)
    chk_path_c = tk.Checkbutton(frm_path, text="Axis C", variable=var_path_c)
    chk_path_c.pack(anchor=tk.W)
    var_path_c.set(False)

    # phase_step
    frm_step = ttk.Labelframe(root, relief="ridge", text="Rotation phase per step(deg)", labelanchor='n')
    frm_step.pack(side="left", fill=tk.Y)

    # var_phase_step = tk.StringVar(root)
    var_phase_step.set(str(phase_step_deg))
    spn_step = tk.Spinbox(
        frm_step, textvariable=var_phase_step, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_phase_step_deg(float(var_phase_step.get())), width=5
    )
    spn_step.pack(side="left")


def create_animation_control():
    frm_anim = ttk.Labelframe(root, relief="ridge", text="Animation", labelanchor="n")
    frm_anim.pack(side="left", fill=tk.Y)
    btn_play = tk.Button(frm_anim, text="Play/Pause", command=switch)
    btn_play.pack(fill=tk.X)
    btn_reset = tk.Button(frm_anim, text="Reset", command=reset)
    btn_reset.pack(fill=tk.X)
    btn_clear = tk.Button(frm_anim, text="Clear path", command=lambda: clear_path())
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
    """
    three_axes.rotate_all_axis_a()
    three_axes.rotate_all_axis_b()
    three_axes.rotate_all_axis_c()
    """
    three_axes.rotate_all_resultant_ab()
    three_axes.rotate_all_resultant_bc()
    three_axes.rotate_all_resultant_ca()

    if var_path_a.get():
        path_a.set_is_draw_path(True)
        path_a.append_path(three_axes.get_axis_a())
    else:
        path_a.set_is_draw_path(False)
    if var_path_b.get():
        path_b.set_is_draw_path(True)
        path_b.append_path(three_axes.get_axis_b())
    else:
        path_b.set_is_draw_path(False)
    if var_path_c.get():
        path_c.set_is_draw_path(True)
        path_c.append_path(three_axes.get_axis_c())
    else:
        path_c.set_is_draw_path(False)


def reset():
    global is_play
    cnt.reset()
    three_axes.reset()


def switch():
    global is_play
    is_play = not is_play


def update(f):
    """
    if var_rot_a.get():
        three_axes.set_scale_a(1)
    else:
        three_axes.set_scale_a(0)
    if var_rot_b.get():
        three_axes.set_scale_b(1)
    else:
        three_axes.set_scale_b(0)
    if var_rot_c.get():
        three_axes.set_scale_c(1)
    else:
        three_axes.set_scale_c(0)
    """
    if is_play:
        cnt.count_up()
        update_diagrams()


""" main loop """
if __name__ == "__main__":
    cnt = Counter(ax=ax0, is3d=True, xy=np.array([x_min, y_max]), z=z_max, label="Step=")
    draw_static_diagrams()
    create_animation_control()
    create_parameter_setter()

    three_axes = ThreeAxes(ax0)

    path_a = Path(ax0, "red")
    path_a.set_is_draw_path(False)

    path_b = Path(ax0, "green")
    path_b.set_is_draw_path(False)

    path_c = Path(ax0, "blue")
    path_c.set_is_draw_path(False)

    ax0.legend(loc='lower right', fontsize=8)

    anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
    root.mainloop()

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
theta_init_deg, phi_init_deg = 90., 0.
rot_velocity_a, rot_velocity_b, rot_velocity_c = 1., 1., 1.

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
ax0 = fig.add_subplot(111, projection='3d')
ax0.set_box_aspect((1, 1, 1))
ax0.grid()
ax0.set_title(title_ax0)
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("t")
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
var_axis_op = tk.IntVar()
var_turn_op = tk.IntVar()
var_theta = tk.StringVar(root)
var_phi = tk.StringVar(root)


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
        # self.size = size
        # self.color = color

        # self.roll_axis_init = np.array([1., 0., 0.])
        # self.pitch_axis_init = np.array([0., 1., 0.])
        # self.yaw_axis_init = np.array([0., 0., 1.])

        self.roll_axis = np.array([1., 0., 0.])
        self.pitch_axis = np.array([0., 1., 0.])
        self.yaw_axis = np.array([0., 0., 1.])

        self.qvr_roll_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                            self.roll_axis[0], self.roll_axis[1], self.roll_axis[2],
                                            length=1, color="red", normalize=True)

        self.qvr_pitch_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                             self.pitch_axis[0], self.pitch_axis[1], self.pitch_axis[2],
                                             length=1, color="blue", normalize=True)

        self.qvr_yaw_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.yaw_axis[0], self.yaw_axis[1], self.yaw_axis[2],
                                           length=1, color="green", normalize=True)

        self.x_path_roll_axis = []
        self.y_path_roll_axis = []
        self.z_path_roll_axis = []
        self.path_roll_axis, = self.ax.plot(np.array(self.x_path_roll_axis),
                                            np.array(self.y_path_roll_axis),
                                            np.array(self.z_path_roll_axis),
                                            color="red", linewidth=1, label="Roll axis")

        self.x_path_pitch_axis = []
        self.y_path_pitch_axis = []
        self.z_path_pitch_axis = []
        self.path_pitch_axis, = self.ax.plot(np.array(self.x_path_pitch_axis),
                                             np.array(self.y_path_pitch_axis),
                                             np.array(self.z_path_pitch_axis),
                                             color="blue", linewidth=1, label="Pitch axis")

        self.x_path_yaw_axis = []
        self.y_path_yaw_axis = []
        self.z_path_yaw_axis = []
        self.path_yaw_axis, = self.ax.plot(np.array(self.x_path_yaw_axis),
                                           np.array(self.y_path_yaw_axis),
                                           np.array(self.z_path_yaw_axis),
                                           color="green", linewidth=1, label="Yaw axis")

        self.r_roll_axis, self.theta_roll_axis, self.phi_roll_axis = cartesian_to_spherical(
            self.roll_axis[0], self.roll_axis[1], self.roll_axis[2])
        self.r_pitch_axis, self.theta_pitch_axis, self.phi_pitch_axis = cartesian_to_spherical(
            self.pitch_axis[0], self.pitch_axis[1], self.pitch_axis[2])
        self.r_yaw_axis, self.theta_yaw_axis, self.phi_yaw_axis = cartesian_to_spherical(
            self.yaw_axis[0], self.yaw_axis[1], self.yaw_axis[2])

        print("row", self.theta_roll_axis, self.phi_roll_axis)

    def roll(self, angle):
        self.roll_axis = self.roll_axis / np.linalg.norm(self.roll_axis)
        rot_matrix = Rotation.from_rotvec(angle * self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self._update_quiver()
        self._update_path()

    def pitch(self, angle):
        self.pitch_axis = self.pitch_axis / np.linalg.norm(self.pitch_axis)
        rot_matrix = Rotation.from_rotvec(angle * self.pitch_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self._update_quiver()
        self._update_path()

    def yaw(self, angle):
        self.yaw_axis = self.yaw_axis / np.linalg.norm(self.yaw_axis)
        rot_matrix = Rotation.from_rotvec(angle * self.yaw_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)

        self._update_quiver()
        self._update_path()

    def rot_x(self, angle):
        rot_matrix = Rotation.from_rotvec(angle * vector_x_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self._update_quiver()
        self._update_path()

    def rot_y(self, angle):
        rot_matrix = Rotation.from_rotvec(angle * vector_y_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self._update_quiver()
        self._update_path()

    def rot_z(self, angle):
        rot_matrix = Rotation.from_rotvec(angle * vector_z_axis)
        self.roll_axis = rot_matrix.apply(self.roll_axis)
        self.pitch_axis = rot_matrix.apply(self.pitch_axis)
        self.yaw_axis = rot_matrix.apply(self.yaw_axis)

        self._update_quiver()
        self._update_path()

    def reset(self):
        self.xyz = np.array([0., 0., 0.])
        self.roll_axis = np.array([1., 0., 0.])
        self.pitch_axis = np.array([0., 1., 0.])
        self.yaw_axis = np.array([0., 0., 1.])

        self.r_roll_axis, self.theta_roll_axis, self.phi_roll_axis = cartesian_to_spherical(
            self.roll_axis[0], self.roll_axis[1], self.roll_axis[2])
        self.r_pitch_axis, self.theta_pitch_axis, self.phi_pitch_axis = cartesian_to_spherical(
            self.pitch_axis[0], self.pitch_axis[1], self.pitch_axis[2])
        self.r_yaw_axis, self.theta_yaw_axis, self.phi_yaw_axis = cartesian_to_spherical(
            self.yaw_axis[0], self.yaw_axis[1], self.yaw_axis[2])

        self.x_path_roll_axis = []
        self.y_path_roll_axis = []
        self.z_path_roll_axis = []

        self.x_path_pitch_axis = []
        self.y_path_pitch_axis = []
        self.z_path_pitch_axis = []

        self.x_path_yaw_axis = []
        self.y_path_yaw_axis = []
        self.z_path_yaw_axis = []

        self._update_quiver()
        self._update_path()

    def _update_quiver(self):
        self.qvr_roll_axis.remove()
        self.qvr_roll_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                            self.roll_axis[0], self.roll_axis[1], self.roll_axis[2],
                                            length=1, color="red", normalize=True)
        self.qvr_pitch_axis.remove()
        self.qvr_pitch_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                             self.pitch_axis[0], self.pitch_axis[1], self.pitch_axis[2],
                                             length=1, color="blue", normalize=True)
        self.qvr_yaw_axis.remove()
        self.qvr_yaw_axis = self.ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                                           self.yaw_axis[0], self.yaw_axis[1], self.yaw_axis[2],
                                           length=1, color="green", normalize=True)

    def _update_path(self):
        self.x_path_roll_axis.append(self.roll_axis[0])
        self.y_path_roll_axis.append(self.roll_axis[1])
        self.z_path_roll_axis.append(self.roll_axis[2])
        self.path_roll_axis.set_xdata(np.array(self.x_path_roll_axis))
        self.path_roll_axis.set_ydata(np.array(self.y_path_roll_axis))
        self.path_roll_axis.set_3d_properties(np.array(self.z_path_roll_axis))

        self.x_path_pitch_axis.append(self.pitch_axis[0])
        self.y_path_pitch_axis.append(self.pitch_axis[1])
        self.z_path_pitch_axis.append(self.pitch_axis[2])
        self.path_pitch_axis.set_xdata(np.array(self.x_path_pitch_axis))
        self.path_pitch_axis.set_ydata(np.array(self.y_path_pitch_axis))
        self.path_pitch_axis.set_3d_properties(np.array(self.z_path_pitch_axis))

        self.x_path_yaw_axis.append(self.yaw_axis[0])
        self.y_path_yaw_axis.append(self.yaw_axis[1])
        self.z_path_yaw_axis.append(self.yaw_axis[2])
        self.path_yaw_axis.set_xdata(np.array(self.x_path_yaw_axis))
        self.path_yaw_axis.set_ydata(np.array(self.y_path_yaw_axis))
        self.path_yaw_axis.set_3d_properties(np.array(self.z_path_yaw_axis))

    def set_theta_initial(self, theta):
        self.theta_roll_axis = theta
        self.theta_yaw_axis = theta - np.pi / 2.

        self.roll_axis[0], self.roll_axis[1], self.roll_axis[2] = (
            spherical_to_cartesian(self.r_roll_axis, self.theta_roll_axis, self.phi_roll_axis))
        self.yaw_axis[0], self.yaw_axis[1], self.yaw_axis[2] = (
            spherical_to_cartesian(self.r_yaw_axis, self.theta_yaw_axis, self.phi_yaw_axis))

        self.pitch_axis = - np.cross(self.roll_axis, self.yaw_axis)

        self._update_quiver()

        self.x_path_roll_axis = []
        self.y_path_roll_axis = []
        self.z_path_roll_axis = []

        self.x_path_pitch_axis = []
        self.y_path_pitch_axis = []
        self.z_path_pitch_axis = []

        self.x_path_yaw_axis = []
        self.y_path_yaw_axis = []
        self.z_path_yaw_axis = []

        self._update_path()

    def set_phi_initial(self, phi):
        self.phi_roll_axis = phi
        self.phi_yaw_axis = phi

        self.roll_axis[0], self.roll_axis[1], self.roll_axis[2] = (
            spherical_to_cartesian(self.r_roll_axis, self.theta_roll_axis, self.phi_roll_axis))
        self.yaw_axis[0], self.yaw_axis[1], self.yaw_axis[2] = (
            spherical_to_cartesian(self.r_yaw_axis, self.theta_yaw_axis, self.phi_yaw_axis))

        self.pitch_axis = - np.cross(self.roll_axis, self.yaw_axis)

        self._update_quiver()

        self.x_path_roll_axis = []
        self.y_path_roll_axis = []
        self.z_path_roll_axis = []

        self.x_path_pitch_axis = []
        self.y_path_pitch_axis = []
        self.z_path_pitch_axis = []

        self.x_path_yaw_axis = []
        self.y_path_yaw_axis = []
        self.z_path_yaw_axis = []

        self._update_path()


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
    btn_play.pack(side="left")
    btn_reset = tk.Button(frm_anim, text="Reset", command=reset)
    btn_reset.pack(side="left")


def set_v_roll(velocity):
    global rot_velocity_a
    rot_velocity_a = velocity


def set_v_pitch(velocity):
    global rot_velocity_b
    rot_velocity_b = velocity


def set_v_yaw(velocity):
    global rot_velocity_c
    rot_velocity_c = velocity


def set_theta_initial(theta):
    three_arrow.set_theta_initial(np.deg2rad(theta))


def set_phi_initial(phi):
    three_arrow.set_phi_initial(np.deg2rad(phi))


def create_parameter_setter():
    global var_axis_op, var_turn_op, var_theta, var_phi
    frm_dir = ttk.Labelframe(root, relief="ridge", text="Initial direction", labelanchor='n')
    frm_dir.pack(side='left', fill=tk.Y)

    lbl_theta = tk.Label(frm_dir, text="Theta")
    lbl_theta.pack(side="left")

    # var_theta = tk.StringVar(root)
    var_theta.set(str(theta_init_deg))
    spn_theta = tk.Spinbox(
        frm_dir, textvariable=var_theta, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_theta_initial(float(var_theta.get())), width=5
    )
    spn_theta.pack(side="left")

    lbl_phi = tk.Label(frm_dir, text="Phi")
    lbl_phi.pack(side="left")

    # var_phi = tk.StringVar(root)
    var_phi.set(str(phi_init_deg))
    spn_phi = tk.Spinbox(
        frm_dir, textvariable=var_phi, format="%.0f", from_=-360, to=360, increment=1,
        command=lambda: set_phi_initial(float(var_phi.get())), width=5
    )
    spn_phi.pack(side="left")

    frm_axis = ttk.Labelframe(root, relief="ridge", text="Rotation axis", labelanchor='n')
    frm_axis.pack(side='left', fill=tk.Y)

    # var_axis_op = tk.IntVar()
    rd_op_axis_rpy = tk.Radiobutton(frm_axis, text="Roll,Pitch,Yaw", value=1, variable=var_axis_op)
    rd_op_axis_rpy.pack(side='left')

    rd_op_axis_xyz = tk.Radiobutton(frm_axis, text="x,y,z", value=2, variable=var_axis_op)
    rd_op_axis_xyz.pack(side='left')

    var_axis_op.set(1)

    frm_v = ttk.Labelframe(root, relief="ridge", text="Rotation velocity", labelanchor='n')
    frm_v.pack(side='left', fill=tk.Y)

    lbl_vr = tk.Label(frm_v, text="A(Roll,x) axis")
    lbl_vr.pack(side="left")

    var_vr = tk.StringVar(root)
    var_vr.set(str(rot_velocity_a))
    spn_vr = tk.Spinbox(
        frm_v, textvariable=var_vr, format="%.0f", from_=-10, to=10, increment=1,
        command=lambda: set_v_roll(float(var_vr.get())), width=5
    )
    spn_vr.pack(side="left")

    lbl_vp = tk.Label(frm_v, text="B(Pitch,y) axis")
    lbl_vp.pack(side="left")

    var_vp = tk.StringVar(root)
    var_vp.set(str(rot_velocity_b))
    spn_vp = tk.Spinbox(
        frm_v, textvariable=var_vp, format="%.0f", from_=-10, to=10, increment=1,
        command=lambda: set_v_pitch(float(var_vp.get())), width=5
    )
    spn_vp.pack(side="left")

    lbl_vy = tk.Label(frm_v, text="C(Yaw,z) axis")
    lbl_vy.pack(side="left")

    var_vy = tk.StringVar(root)
    var_vy.set(str(rot_velocity_c))
    spn_vy = tk.Spinbox(
        frm_v, textvariable=var_vy, format="%.0f", from_=-10, to=10, increment=1,
        command=lambda: set_v_yaw(float(var_vy.get())), width=5
    )
    spn_vy.pack(side="left")

    frm_turn = ttk.Labelframe(root, relief="ridge", text="Turn of rotation", labelanchor='n')
    frm_turn.pack(side='left', fill=tk.Y)

    # var_turn_op = tk.IntVar()
    rd_op_rpy = tk.Radiobutton(frm_turn, text="A->B->C", value=1, variable=var_turn_op)
    rd_op_rpy.pack(side='left')

    rd_op_pyr = tk.Radiobutton(frm_turn, text="C->B->A", value=2, variable=var_turn_op)
    rd_op_pyr.pack(side='left')

    var_turn_op.set(1)


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
    # angle = np.deg2rad(cnt.get()) % (2. * np.pi)
    angle = np.deg2rad(1)
    if var_axis_op.get() == 1:
        if var_turn_op.get() == 1:
            # Roll->Pitch->Yaw
            three_arrow.roll(rot_velocity_a * angle)
            three_arrow.pitch(rot_velocity_b * angle)
            three_arrow.yaw(rot_velocity_c * angle)
        else:
            # Yaw->Pitch->Roll
            three_arrow.yaw(rot_velocity_c * angle)
            three_arrow.pitch(rot_velocity_b * angle)
            three_arrow.roll(rot_velocity_a * angle)
    else:
        if var_turn_op.get() == 1:
            # x->y->z
            three_arrow.rot_x(rot_velocity_a * angle)
            three_arrow.rot_y(rot_velocity_b * angle)
            three_arrow.rot_z(rot_velocity_c * angle)
        else:
            # z->y->x
            three_arrow.rot_z(rot_velocity_c * angle)
            three_arrow.rot_y(rot_velocity_b * angle)
            three_arrow.rot_x(rot_velocity_a * angle)


def reset():
    global is_play, var_theta
    is_play = False
    cnt.reset()
    three_arrow.reset()
    var_theta.set(str(theta_init_deg))
    var_phi.set(str(phi_init_deg))


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

    ax0.legend(loc='lower right', fontsize=8)

    anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
    root.mainloop()

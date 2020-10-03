import taichi as ti
import os

def animate(xs, goals, gravitations, gravitation_position, outputdir):
    assert len(xs) == len(gravitations)
    gui = ti.GUI("Electric", (512, 512), background_color=0x3C733F, show_gui=False)

    for t in range(len(xs)):
        gui.clear()

        for i in range(len(gravitation_position)):
            r = (gravitations[t][i] + 1) * 30
            gui.circle(gravitation_position[i], 0xccaa44, r)

        gui.circle((xs[t][0], xs[t][1]), 0xF20530, 30)
        gui.circle((goals[t][0], goals[t][1]), 0x3344cc, 10)

        gui.show(os.path.join(outputdir, "{:04d}.png".format(t + 1)))

import math
import importlib

def generate_hexagonal_grid(boundingbox, spacing, compare_radius):
    """Generates an hexagonal grid inside a given bounding-box with a given spacing between the vertices"""
    hexheight = spacing
    hexwidth = math.sqrt(3) * spacing / 2

    # for debug
    assert ( compare_radius < int(hexwidth/2) )

    vertspacing = 0.75 * hexheight
    horizspacing = hexwidth
    sizex = int((boundingbox[1] - boundingbox[0]) / horizspacing) 
    sizey = int(((boundingbox[3] - boundingbox[2]) - hexheight) / vertspacing) + 1
    # if sizey % 2 == 0:
    #     sizey += 1
    pointsret = []
    for i in range(0, sizex):
        for j in range(0, sizey):
            xpos = int(i * horizspacing + horizspacing/2)
            ypos = int(j * vertspacing + hexheight/2)
            if j % 2 == 1:
                xpos += int(horizspacing * 0.5)
            if (xpos>boundingbox[1]) or ((xpos+compare_radius) > boundingbox[1] ) or ((ypos+compare_radius) > boundingbox[3]):
                continue
            assert int(xpos + boundingbox[0]) < boundingbox[1]
            pointsret.append([int(xpos + boundingbox[0]), int(ypos + boundingbox[2])])
    print('\n\nboundingbox for mesh is:{}'.format(boundingbox))
    print('the last element of meshbox is:{}\n'.format(pointsret[-1]))
    return pointsret

def load_plugin(class_full_name):
    package, class_name = class_full_name.rsplit('.', 1)
    plugin_module = importlib.import_module(package)
    plugin_class = getattr(plugin_module, class_name)
    return plugin_class

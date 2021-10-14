import numpy as np
import h5py
from . import surfaces
from . import curves as crvs

def readTagData(filename):
    f = h5py.File(filename, 'r')
    fkeys = list(f.keys())
    if 'Template' in fkeys:
        vert = np.array(f['Template']['Vertices']).T
        faces = np.array(f['Template']['Faces'], dtype=int).T -1
        Template = surfaces.Surface(surf=(faces, vert))
    else:
        Template = None

    if 'Tagplane' not in fkeys or 'Curves' not in fkeys:
        print('Missing information in ' + filename)
        return None, None, None

    tag_pl = np.array(f['Tagplane']['Corners']).T
    npl = tag_pl.shape[0]

    g = f['Curves']
    phases = list(g.keys())
    curves = {}
    for ph in phases:
        images = list(g[ph].keys())
        nim = len(images)
        curve_in_phase = {s: None for s in images}
        for im in images:
            h = g[ph][im]
            curve_in_image = {}
            #curve_in_image = {'Intersections' + str(k): None for k in range(npl)}
            intersections = list(h.keys())
            # if len(intersections) != npl:
            #     print('A possibly empty intersection must be provided for each tag plane')
            #     return None, None, None
            for ins in intersections:
                if len(h[ins]) > 0:
                    vert = np.array(h[ins]['Vertices']).T
                    faces = np.array(h[ins]['Edge'], dtype=int).T -1
                    curve_in_image[ins] = crvs.Curve(curve=(faces,vert))
            curve_in_phase[im] = curve_in_image
        curves[ph] = curve_in_phase

    activity = f['ActiveMatrix']
    return Template, tag_pl, curves, activity

def writeTagData(filename, tag_pl, curves, Template = None):
    f = h5py.File(filename, 'w')
    if Template is not None:
        f.create_group('Template')
        f['Template'].create_dataset('Vertices', data=Template.vertices)
        f['Template'].create_dataset('Faces', data=Template.faces)

    f.create_dataset('Tag_planes', data=tag_pl)

    f.create_group('Curves')
    g = f['curves']
    for k,ph in enumerate(curves):
        g.create_group(f'Phase{k+1}')
        h = g[f'Phase{k+1}']
        for l,inter in enumerate(ph):
            if inter is not None:
                h.create_dataset('Vertices', data=inter.vertices)
                h.create_dataset('Edges', data=inter.faces)


<<<<<<< HEAD
=======
#readTagData('/Users/younes/Johns Hopkins/Tag MRI Research - Data/HDF5_SA50_43R_20180102_PRE-2018-01-05_JW/SA50_43R_20180102_PRE-2018-01-05_JW.h5')
>>>>>>> 03d4865ab4511f87e67a43016abeafbe6fcb5c59

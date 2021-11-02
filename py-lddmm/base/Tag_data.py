import numpy as np
import h5py
from . import surfaces
from . import curves as crvs
from . import surfaceSection

def readTagData(filename, contoursOnly=False):
    f = h5py.File(filename, 'r')
    fkeys = list(f.keys())
    if not contoursOnly:
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

    epi_contours = []
    endo_contours = []
    if 'Epi' in fkeys:
        images = list(f['Epi'].keys())
        nphases = f['Epi'][images[0]].shape[0]
        epi_contours = [[]] * nphases
        for img in f['Epi']:
            allc = f['Epi'][img]
            for i in range(allc.shape[0]):
                c = crvs.Curve(curve=allc[i,:,:].T)
                c = surfaceSection.SurfaceSection(curve=c)
                epi_contours[i].append(c)
    if 'Endo' in fkeys:
        images = list(f['Endo'].keys())
        nphases = f['Endo'][images[0]].shape[0]
        endo_contours = [[]] * nphases
        for img in f['Endo']:
            allc = f['Endo'][img]
            for i in range(allc.shape[0]):
                c = crvs.Curve(curve=allc[i,:,:].T)
                c = surfaceSection.SurfaceSection(curve=c)
                endo_contours[i].append(c)

    contours = {}
    contours['Epi'] = epi_contours
    contours['Endo'] = endo_contours

    if contoursOnly:
        return contours
    else:
        return Template, tag_pl, curves, activity, contours

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



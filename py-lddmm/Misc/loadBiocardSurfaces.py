import Surfaces

def load():
    #path = '/Volumes/project/biocard/data/phase_1_surface_mapping_new_structure/'
    path = '/Users/younes/MorphingData/Biocard/'
    sub1 = '0186193_1_L'
    #sub2 = '1449400_1_L'
    sub2 = '2105688_1_L'
    f0 = []
    f0.append(Surfaces.Surface(filename =path + 'hippocampus/' + sub1 + '_qc.byu'))
    f0.append(Surfaces.Surface(filename =path + 'amygdala/' + sub1 + '_qc.byu'))
    f1 = []
    f1.append(Surfaces.Surface(filename =path + 'hippocampus/' + sub2 + '_qc.byu'))
    f1.append(Surfaces.Surface(filename =path + 'amygdala/' + sub2 + '_qc.byu'))
    return f0, f1

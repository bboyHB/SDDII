import os
import shutil

DAGM_path = 'DAGM'
classes = os.listdir(DAGM_path)

for cls in classes:
    output_path = os.path.join(f'DAGM_{cls}_cycle')
    trainA_path = os.path.join(output_path, 'trainA')
    trainB_path = os.path.join(output_path, 'trainB')
    testA_path = os.path.join(output_path, 'testA')
    testB_path = os.path.join(output_path, 'testB')
    os.makedirs(trainA_path)

    os.makedirs(trainB_path)
    os.makedirs(testA_path)
    os.makedirs(testB_path)
    for dir in ['Train', 'Test']:
        pic_names = {name for name in os.listdir(os.path.join(DAGM_path, cls, dir)) if name.endswith('PNG')}
        defect_names = {name[:4] + name[-4:] for name in os.listdir(os.path.join(DAGM_path, cls, dir, 'Label')) if name.endswith('PNG')}
        normal_names = pic_names - defect_names
        destA = eval(dir.lower() + 'A_path')
        destB = eval(dir.lower() + 'B_path')
        for normal in normal_names:
            shutil.copy(os.path.join(DAGM_path, cls, dir, normal), os.path.join(destA, normal.replace('PNG', 'png')))
        for defect in defect_names:
            shutil.copy(os.path.join(DAGM_path, cls, dir, defect), os.path.join(destB, defect.replace('PNG', 'png')))


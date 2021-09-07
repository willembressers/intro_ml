import os
import shutil

source = os.path.join('..', 'notebooks', 'experiment', 'excercise.ipynb')

for i, subject_i in enumerate(['classification', 'regression', 'clustering', 'dimensionality_reduction']):

    for j, subject_j in enumerate(['1', '2', '3']):
        
        # copy example file
        destination = os.path.join('..', 'notebooks', 'report', 'exercises', f'{i+1}.{j+1}-{subject_i}.ipynb')
        shutil.copy2(source, destination)
        
        # replace file contents
        with open(destination, 'r') as file :
            filedata = file.read()
        filedata = filedata.replace("# Exercise: 0.0", f"# Exercise: {i+1}.{j+1} {subject_i}")
        filedata = filedata.replace('0.0.csv', f'{i+1}.{j+1}.csv')
        with open(destination, 'w') as file:
            file.write(filedata)
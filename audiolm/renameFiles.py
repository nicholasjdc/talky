import os

objPath = os.getcwd()
relPath = '/results'
absPath = objPath + relPath

for filename in os.listdir(absPath):
    splitFile = filename.split('.')
    if splitFile[0] == 'fine':
        newFilename = filename.replace('fine', 'coarse')
        os.rename(absPath+'/'+filename, absPath +'/'+newFilename)

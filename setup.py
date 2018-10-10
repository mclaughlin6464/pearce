from distutils.core import setup
from glob import glob
#scripts = glob('bin/*/*')

setup(name='pearce',
        version='0.1',
        description='Pratical Emulation for Analysis and Regression of Cosmological Enviornments.',
        author='Sean McLaughlin',
        author_email='swmclau2@stanford.edu',
        url='https://github.com/mclaughlin6464/pearce',
        #scripts=scripts,
        packages=['pearce', 'pearce.emulator', 'pearce.mocks','pearce.inference', 'pearce.mocks.assembias_models'])

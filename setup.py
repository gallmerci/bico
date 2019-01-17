from setuptools import setup

setup(
    name='bico',
    version='0.2',
    packages=['bico', 'bico.geometry', 'bico.nearest_neighbor', 'bico.utils'],
    url='https://github.com/gallmerci/bico',
    license='MIT',
    author='Marc Bury',
    author_email='burycram@googlemail.com',
    description='BICO is a fast streaming algorithm and reduction technique for the k-means problem.',
    install_requires=['numpy', 'scipy', 'nearpy', 'attrs']
)

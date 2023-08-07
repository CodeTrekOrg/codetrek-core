from setuptools import setup, find_packages

setup(
    name='codetrek',
    version='0.1',
    packages=find_packages(),
    install_requires=['torch', 'numpy', 'tqdm', 'scikit-learn'],
    author='Pardis Pashakhanloo',
    author_email='ppashakhanloo@gmail.com',
    description='CodeTrek Training and Preprocessing Module',
    url='https://github.com/CodeTrekOrg/codetrek-core',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
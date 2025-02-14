from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)   
    return requirements

#just update
setup(
    name='Fault detection',
    version='0.0.1',
    author='Rakshit',
    author_email='rakshit2004gupta@gmail.com',
    install_requires=get_requirements('requirements.txt'),  # External dependencies only
    packages=find_packages()
)

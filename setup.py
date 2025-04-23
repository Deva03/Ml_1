from setuptools import find_packages,setup
from typing import List

Hyphen_e_dot = "-e ."

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    :param file_path:
    :return:
    '''

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [requirement.replace("\n","") for requirement in requirements]

        if Hyphen_e_dot in requirements :
            requirements.remove(Hyphen_e_dot)

    return requirements


setup(
    name="Ml_1",
    version='0.0.1',
    author="devamshani",
    author_email="devamshani777@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
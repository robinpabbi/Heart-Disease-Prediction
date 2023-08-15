from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT="-e ."

def get_dependencies(filePath:str)->List:
    required_libraries = []
    with open(filePath) as fileObj:
        required_libraries = fileObj.readlines()
        required_libraries = [reqLib.replace("\n", "") for reqLib in required_libraries]

        if HYPHEN_E_DOT in required_libraries:
            required_libraries.remove(HYPHEN_E_DOT)

    return required_libraries       


setup(
    name= "HeartDiseasePrediction",
    version="0.0.1",
    description="This project predicts posibility of heart disease based on ",
    author="Robin Pabbi",
    author_email="",
    install_requires=get_dependencies('requirements.txt'),
    packages=find_packages()
)
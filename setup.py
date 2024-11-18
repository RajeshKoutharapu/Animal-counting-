from setuptools import setup, find_packages

setup(
    name='sheep_detection_project',
    version='1.0.0',
    description='A YOLO-based object detection system for tracking sheep in videos and ROS camera streams.',
    author='Your Name',
    python_requires='>=3.7',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[line.strip() for line in open('requirements.txt')],
)

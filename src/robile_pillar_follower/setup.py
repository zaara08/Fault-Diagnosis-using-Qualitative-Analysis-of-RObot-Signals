from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robile_pillar_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alisha',
    maintainer_email='alishasyedkarimulla@gmail.com',
    description='Robile pillar wall follower with data logger',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pillar_follower_node = '
            'robile_pillar_follower.pillar_follower_node:main',
            'data_logger_node = '
            'robile_pillar_follower.data_logger_node:main',
        ],
    },
)
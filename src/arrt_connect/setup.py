from setuptools import setup

package_name = 'arrt_connect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    py_modules=[
        'search',
        'arrt_connect.arrt_connect_node',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='mdevhurt@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arrt_connect_node = arrt_connect.arrt_connect_node:main'
        ],
    },
)

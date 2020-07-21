from setuptools import setup

package_name = 'trt_yolov3'

setup(
    name=package_name,
    version='0.1.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	('lib/' + package_name, ['trt_yolov3/yolov3-416.trt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artiv',
    maintainer_email='mail',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trt_yolov3_node = trt_yolov3.trt_yolov3_node:main'
        ],
    },
)

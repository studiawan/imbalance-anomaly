from setuptools import setup

setup(name='imbalance_anomaly',
      version='0.0.1',
      description='Anomaly detection in imbalance authentication logs.',
      long_description='Anomaly detection in imbalance authentication logs.',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
      ],
      keywords='anomaly detection',
      url='http://github.com/studiawan/imbalance-anomaly/',
      author='Hudan Studiawan',
      author_email='studiawan@gmail.com',
      license='MIT',
      packages=['imbalance_anomaly'],
      install_requires=[
          'scikit-learn==0.22.1',
          'keras==2.2.4',
          'tensorflow-gpu==1.15.2',
          'keras-metrics==1.1.0',
          'imbalanced-learn==0.6.1'
      ],
      include_package_data=True,
      zip_safe=False)

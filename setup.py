from setuptools import setup

setup(name='imbalance-anomaly',
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
      packages=['imbalance-anomaly'],
      install_requires=[
          'scikit-learn',
          'keras',
          'tensorflow-gpu',
          'keras-metrics',
          'imbalanced-learn'
      ],
      include_package_data=True,
      zip_safe=False)

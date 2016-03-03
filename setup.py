from setuptools import setup

setup(name='tensorprob',
      version='0.0.0',
      description='A probabilistic graphical modeling framework based on Tensorflow',
      url='http://ibab.github.io/tensorprob',
      author='Igor Babuschkin',
      author_email='igor@babuschk.in',
      license='MIT',
      install_requires=[
          'tensorflow',
      ],
      packages=[
          'tensorprob',
          'tensorprob.distributions',
          'tensorprob.optimizers',
          'tensorprob.samplers'
      ],
      zip_safe=False)

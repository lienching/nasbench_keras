from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages    

def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_deps = ['numpy', 'tensorflow']
if get_dist('tensorflow') is None and get_dist('tensorflow-gpu') is not None:
    install_deps.remove('tensorflow')

setup(
  name = 'nasbench_keras',         # How you named your package folder (MyLib)
  packages = ['nasbench_keras'],   # Chose the same as "name"
  version = '0.2',      # Start with a small number and increase it with every change you make
  license='MIT License',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'NAS-Bench 101 to Tensorflow 2.0 (tf.keras) converter',   # Give a short description about your library
  author = 'evgps',                   # Type in your name
  author_email = 'evgps@ya.ru',      # Type in your E-Mail
  url = 'https://github.com/evgps/nasbench_keras',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/evgps/nasbench_keras/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['NAS', 'Converter', 'Tensorflow', 'Keras'],   # Keywords that define your package best
  install_requires=install_deps,
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3'      #Specify which pyhton versions that you want to support
  ],
)
from setuptools import setup

setup(
    name='event_predictor',
    version='0.1',
    packages=['Scorer', 'Trainer'],
	install_requires=['flask', 'sklearn', 'pandas'],
    url='https://github.com/Ben-Pollard/event_predictor',
    license='',
    author='Ben-Pollard',
    author_email='',
    description='Predicts popularity of tech meetups'
)

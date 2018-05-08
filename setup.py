from setuptools import setup

setup(name='habits',
      version='0.1',
      description='Welcome my son. Welcome to the habits machine.',
      url='https://github.com/nitinvwaran/SpeechRecognition',
      author='Nitin Venkateswaran',
      author_email='nitin.vwaran@gmail.com',
      license='MIT',
      packages=['habits'],
      install_requires=['numpy','python_speech_features','scipy','tensorflow'],
      scripts=['bin/process_habit'],
      zip_safe=False)
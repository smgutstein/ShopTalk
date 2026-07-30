from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='shoptalk',
    version='0.0.1',
    packages=find_packages(),
    description='Multimodal RAG Shopping Chatbot',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/smgutstein/ShopTalk',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    install_requires=required,
)
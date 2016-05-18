Install virtualenv: https://virtualenvwrapper.readthedocs.io/en/latest/

Setup:
cd $WORKON_HOME
tar -xf tu_anh.tar.gz
mkvirtualenv tu_anh
cdvirtualenv
pip install -r ./requirements.txt
brew tap homebrew/science
brew install opencv3 --with-contrib
sudo ln -s /usr/local/Cellar/opencv3/3.1.0_1/lib/python2.7/site-packages/cv2.so lib/python2.7/site-packages/cv2.so

Start server:
python manage.py runserver

Go to http://127.0.0.1:8000
cd ~/git
git init --bare tu_anh.git
cd ~/envs
git clone bang@mac.local:/Users/bang/git/tu_anh.git
mkvirtualenv tu_anh
cdvirtualenv
pip install -r ./requirements.txt
brew tap homebrew/science
brew install opencv3 --with-contrib
sudo ln -s /usr/local/Cellar/opencv3/3.1.0_1/lib/python2.7/site-packages/cv2.so lib/python2.7/site-packages/cv2.so

Start server:
python manage.py runserver

Uninstall opencv3
brew uninstall opencv3
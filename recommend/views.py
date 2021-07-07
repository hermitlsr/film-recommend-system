from recommend import app
from flask import render_template
from recommend.main.models import rmdall,topn

from flask import Flask, request, redirect, url_for
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from recommend.main.userlogin import User, get_user
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm

app2 = Flask(__name__)
app.secret_key = '1234567'

login_manager = LoginManager()  # 实例化登录管理对象
login_manager.login_view = 'login'  # 设置用户登录视图函数 endpoint
login_manager.login_message_category = 'info'
login_manager.login_message = '请登录'
login_manager.init_app(app)  # 初始化应用


@login_manager.user_loader  # 定义获取登录用户的方法
def load_user(user_id):
    return User.get(user_id)


@app.route('/')
@app.route('/index/')
@login_required
def index():
    current_userid = int(current_user.username)
    res = rmdall.rec_all(userId=current_userid, n1=10, n2=50, n3=50)
    infor = rmdall.findallmovie(res)
    return render_template('index.html', username=current_userid, items=infor)


# ...
class LoginForm(FlaskForm):
    """登录表单类"""
    username = StringField('用户名', validators=[DataRequired()])
    password = PasswordField('密码', validators=[DataRequired()])


# ...
@app.route('/login/', methods=('GET', 'POST'))  # 登录
def login():
    form = LoginForm()
    emsg = None
    if form.validate_on_submit():
        user_name = form.username.data
        password = form.password.data
        user_info = get_user(user_name)  # 从用户数据中查找用户记录
        if user_info is None:
            emsg = "用户名或密码密码有误"
        else:
            user = User(user_info)  # 创建用户实体
            if user.verify_password(password):  # 校验密码
                login_user(user)  # 创建用户 Session
                return redirect(request.args.get('next') or url_for('index'))
            else:
                emsg = "用户名或密码密码有误"
    return render_template('login.html', form=form, emsg=emsg)


@app.route('/logout')  # 登出
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/topnfilm/')  #
@login_required
def topnfilm():
    res = topn(n=10, tag='all', indate="2018-05-12", days=1080)
    infor = rmdall.findallmovie(res)
    return render_template('topnfilm.html', items=infor)


@app.route('/LFMfilm/')  #
@login_required
def LFMfilm():
    current_userid = int(current_user.username)
    res = rmdall.rec(current_userid, 50)
    infor = rmdall.findallmovie(res)
    return render_template('LFMfilm.html', items=infor)


@app.route('/baseuserfilm/')  #
@login_required
def baseuserfilm():
    current_userid = int(current_user.username)
    res = rmdall.mx(id=current_userid, n=50)
    infor = rmdall.findallmovie(res)
    return render_template('baseuserfilm.html', username=current_userid, items=infor)

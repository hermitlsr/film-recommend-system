from flask_login import UserMixin
import pandas as pd


def read_user():
    user = pd.read_csv("./recommend/data/user.csv")
    users = []
    for id in user['id']:
        userdict = {}
        userdict['id'] = str(id)
        userdict['username'] = str(list(user[user['id'] == id]['username'])[0])
        userdict['password'] = str(list(user[user['id'] == id]['password'])[0])
        users.append(userdict)
    return users


users = read_user()


def get_user(user_name):
    """根据用户名获得用户记录"""

    for user in users:
        if user_name == user['username']:
            return user
    return None


class User(UserMixin):
    """用户类"""

    def __init__(self, user):
        self.id = user.get("id")
        self.username = user.get("username")
        self.password = user.get("password")

    def verify_password(self, password):
        """密码验证"""
        if self.password is None:
            return False
        elif self.password != password:
            return False
        return True

    def get_id(self):
        """获取用户ID"""
        return self.id

    @staticmethod
    def get(user_id):
        """根据用户ID获取用户实体，为 login_user 方法提供支持"""
        if not user_id:
            return None
        for user in users:
            if user.get('id') == user_id:
                return User(user)
        return None

import ldap3

class Ldap:
    """Class for LDAP related connections/operations."""

    def __init__(self, server_uri, ldap_user, ldap_pass):
        self.server = ldap3.Server(server_uri, get_info=ldap3.ALL)
        print(f"Connected to ldap server: {self.server}")
        self.conn = ldap3.Connection(self.server, user=ldap_user, password=ldap_pass, auto_bind=True)

    def who_am_i(self):
        return self.conn.extend.standard.who_am_i()

    
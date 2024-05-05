import ldap3
from ldap3 import SUBTREE

class Ldap:
    """Class for LDAP related connections/operations."""

    def __init__(self, server_uri, ldap_user, ldap_pass):
        self.server = ldap3.Server(server_uri, get_info=ldap3.ALL)
        self.conn = ldap3.Connection(self.server, user=ldap_user, password=ldap_pass, auto_bind=True)

    def who_am_i(self):
        account = self.conn.extend.standard.who_am_i()
        account = account.split('\\')[1]
        return account
    
    def get_department(self, user):
        attributes = ['cn', 'givenName','sAMAccountName', 'department']
        filter = f"(&(objectclass=person)(objectclass=user)(sAMAccountName={user}))"
        result = self.conn.search('ou=GLOBAL IME BANK LIMITED,dc=gibl,dc=org', filter, search_scope=SUBTREE, attributes=attributes)
        if result:
            department = [entry.department.value for entry in self.conn.entries ]
            return department
        else:
            return 


    
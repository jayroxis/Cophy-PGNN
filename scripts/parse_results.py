import re

def parse_results(path):
    f = open(path, 'r')
    lines = list(f.readlines())

    d = "".join(lines)
    d = d.replace('\n', '')
    d = d.replace(' ', '')
    d = d.replace(':<', ':"<')
    d = d.replace('>,', '>",')
    d = d.replace(': nan,', ': "nan",')
    d = re.sub(r"('\:)(\w*\w\()", r'\1"\2', d)
    d = re.sub(r"(\))(\},')", r'\1"\2', d)
    d = re.sub(r"(\w\))(,')", r'\1"\2', d)
    f.close()
    return eval(d)
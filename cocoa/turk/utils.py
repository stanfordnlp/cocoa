from boto.mturk.connection import MTurkConnection
import boto.mturk.qualification as mtqual

def get_mturk_connection(config, debug=False):
    """Connect to MTurk account.

    Args:
        config (dict): {'access_key': str, 'secret_key': str}
        debug (bool): if true, use sandbox

    Returns:
        MTrukConnection

    """
    if debug:
        host = 'mechanicalturk.sandbox.amazonaws.com'
    else:
        host = 'mechanicalturk.amazonaws.com'

    mturk_connection = MTurkConnection(aws_access_key_id=config["access_key"],
                                       aws_secret_access_key=config["secret_key"],
                                       host=host)
    return mturk_connection

def default_qualifications():
    quals = mtqual.Qualifications()
    quals.add(mtqual.LocaleRequirement("EqualTo", "US"))
    quals.add(mtqual.PercentAssignmentsApprovedRequirement("GreaterThan", 95))
    quals.add(mtqual.NumberHitsApprovedRequirement("GreaterThan", 10))
    return quals

def xml_safe(string):
    string = string.replace("&", "&amp;")
    string = string.replace("<", "&lt;")
    string = string.replace(">", "&gt;")
    string = string.replace("\"", "\\\"")
    return string

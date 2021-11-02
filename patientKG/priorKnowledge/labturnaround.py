from ..config.bedrock_connection import *
import pandas as pd
class LabTurnAround:
    def __init__(self, recalculate = False):
        self.turnaround_reference = {}

    def get_reference_from_db(self):
        Red004_Conn = Red004()
        DF = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Lab_Turnaround_Statistics]',Red004_Conn)
        return DF
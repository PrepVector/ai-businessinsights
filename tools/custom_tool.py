from crewai.tools import BaseTool

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

class PythonREPLTool(BaseTool):
    """
    Custom tool to execute Python code and return the result.
    """
    name: str = "PythonREPL"
    description: str = (
        "Executes Python code and returns output."
    )
    
    def _run(self, code: str) -> str:
        """
        Runs the provided Python code and returns the output.
        
        Args:
        code (str): Python code to execute.

        Returns:
        str: Output of the executed code or error message.
        """
        try:
            # Use exec for running the code within a safe scope
            # Create a dictionary to store the execution environment
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import statsmodels.api as sm
            from scipy import stats
            local_scope = {}
            #exec_globals = { "matplotlib": matplotlib, "scipy": scipy, "pandas": pandas, "pd": pd, "np": np, "sns": sns, "plt": plt, "pearsonr":pearsonr} 
            exec(code,globals(),local_scope)  # Execute the code
            return str(local_scope.get('result', 'Execution completed.'))
        except Exception as e:
            return f"Error executing code: {str(e)}"

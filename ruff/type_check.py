import sys
sys.path.append('/home/ubuntu/nextgen')
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    import pandas as pd
import pandas as pd

# This should be valid, no string quotes around pd.DataFrame
def test_func(data) -> Optional[pd.DataFrame]:
    return None

print('Type check successful')

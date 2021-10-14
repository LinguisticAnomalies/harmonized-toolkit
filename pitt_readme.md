THESE NOTES REFER TO THE DATA SPREADSHEET (data.xlsx) FROM THE PITT CORPUS

Column G:  base dx = v-0 dx
Column H:  dx1 = v-1 dx
etc.

Column K:  changeddx = change date
Column L:  interdx = dx changed but then may change again
Columns Q-T:  curdx1-4 = all the dx the person had at last visit

Diagnostic codes:

1, 100 = probable AD
2, 200 = possible AD + something else
3, 300 = vascular dementia
4 = other dementia
    420, 430 = Parkinson
5 = person complains of problems, none diagnosed
    596 - Possible Dementia with Lewy bodies
6, 7 = mild cognitive impairment
       610, 611 = MCI multi cognitive domain type, touch of vis-sp, lang, and mem
       720, 740 = MCI with memory only, all else nl
       730 = psych -- general anxiety, depression no other symptoms, etc.
       770 - presence of cerebrovascular disease, usually small infarcts 
             in the brain detected by MRI, but without history of clinical strokes
8, 800 = control
         851 - control with cerebrovascular disease
9 = olivopontocerebellar degeneration, normal cognition

Column W:  sex, 1=male, 0=female

Tests:
Mini Mental State Exam -- mms
Clinical Dementia Rating Scale -- cdr
Blessed Dementia Scale -- blessed (or bless#)
Hamilton Depression Rating Scale -- Hamilton
Hachinski Ischemic Scale -- htotal (or hrs#total), hmtotal (or hrs#m)
Mattis Dementia Rating Scale -- mattis
NYU Rating Scale (Parkinson's measure) -- nyu
Hachinski Total and modified


Last contact 2000

NOTES ON CHAT FILENAMES

The first first 3 digits correspond to the id numbers in column A of the Data Spreadsheet. 

The number following the dash corresponds to the visit number: 0 (baseline), 1, 2, 3, and 4.  On the Data Spreadsheet, these numbers are entered as 1 (baseline), 2, 3, 4, and 5.  So, 001-0.cha corresponds to visit 1 data on the spreadsheet for id #1; 001-1.cha corresponds to visit 2 data on the spreadsheet for id #1, etc.

NOTES ON CHAT FILE ID TIER INFO

@ID:	eng|UPMC|PAR|age|gender|Dx|002-0-002v-3|Participant|MMSE|

Age is approximate.  Data files give age in years at first visit and then give dates of future testing, but no date of birth is given, so ages are approximated for everything past v=0 and can only be considered accurate within a few months to a year (max).

Dx:  ProbableAD, PossibleAD, MCI, Memory, Vascular, Control

ADRC spreadsheet codes were converted to Diagnostic labels on the ID tier as follows:
600 - MCI
720 and 740 - Memory
101 -  ProbableAD
300 - Vascular
821 - control
500, 540 - Other
Note:  Only main diagnosis was entered into the CHAT file ID tier
from __future__ import unicode_literals, print_function, division
import re
import urllib
import pandas as pd
from warnings import filterwarnings, warn
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.feature_extraction.text import CountVectorizer

__doc___ = """The basic NLP tools needed for the PYQAE toolset, including
parsing RTF data, basic word tokenizer and feature extraction,
plus stop-words for german"""

_dswl_list = 'https://gist.githubusercontent.com/kmader/bb889170010d4b9c90a4e7f66107b94b/raw/d3df37bd770d86a60f1250e675ffd6948f7bf7cc/stop_words.txt'

with urllib.request.urlopen(_dswl_list) as resp:
    deutsch_stop_words = resp.read().decode().split(',')
    ascii_de_stop_words = [strip_accents_ascii(x) for x in deutsch_stop_words]

def _check_de_stop_words():
    """
    >>> len(ascii_de_stop_words)
    1803
    >>> ascii_de_stop_words[998]
    'mehrmaligem'
    """
    pass


def create_word_matrix(in_df, txt_col, stop_words = ascii_de_stop_words):
    """
    Creates a matrix of words with their respective counts and appends it to the existing DataFrame
    :param in_df:
    :param txt_col:
    :param stop_words: words to ignore for the analysis
    :return: DataFrame with words and counts
    >>> import pandas as pd
    >>> create_word_matrix(pd.DataFrame([{'txt': 'hi hi tom'}]), 'txt', [])
             txt  hi  tom
    0  hi hi tom   2    1
    >>> _rows = [{'txt': 'hi hi tom'}, {'txt': 'hi kevin'}]
    >>> create_word_matrix(pd.DataFrame(_rows), 'txt', ['hi'])
             txt  kevin  tom
    0  hi hi tom    0.0  1.0
    1   hi kevin    1.0  0.0
    """
    cv = CountVectorizer(analyzer='word', stop_words = stop_words)
    cv.fit(in_df[txt_col].values)
    new_vocab_dict = {id: word for word, id in cv.vocabulary_.items()}
    t_mat = cv.transform(in_df[txt_col].values)
    row_word_list = [
        {new_vocab_dict[j]: row_vec[i, j] for i, j in zip(*row_vec.nonzero())}
        for row_vec in t_mat]
    row_word_df = pd.DataFrame.from_dict(row_word_list).fillna(0)
    full_word_df = pd.concat([in_df.reset_index(drop=True), row_word_df],
                             axis=1)
    return full_word_df


def to_num_stage(in_str):
    # type: (str) -> int
    """
    Converts a string stage code into a numerical stage
    Examples
    ---

    >>> to_num_stage('2a')
    2
    >>> to_num_stage('3')
    3
    >>> filterwarnings("ignore")
    >>> to_num_stage('x')
    -1
    """
    in_str = str(in_str)
    new_str = in_str[:1]
    try:
        return int(new_str)
    except:
        warn('Cannot convert {} to a numerical stage'.format(in_str),
             RuntimeWarning)
        return -1

def strip_rtf(text):
    # type: (str) -> str
    """
    Remove RTF characters from the code
    Modified from:
        http://stackoverflow.com/a/188877, Code created by Markus Jarderot:
        http://mizardx.blogspot.com

    :param text: str
        raw rtf formatted text
    :return: str
        the text with all RTF formatting tags removed
    >>> strip_rtf('Hello')
    'Hello'

    """
    pattern = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})?[ ]?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)", re.I)
    # control words which specify a "destination".
    destinations = frozenset((
      'aftncn','aftnsep','aftnsepc','annotation','atnauthor','atndate','atnicn','atnid',
      'atnparent','atnref','atntime','atrfend','atrfstart','author','background',
      'bkmkend','bkmkstart','blipuid','buptim','category','colorschememapping',
      'colortbl','comment','company','creatim','datafield','datastore','defchp','defpap',
      'do','doccomm','docvar','dptxbxtext','ebcend','ebcstart','factoidname','falt',
      'fchars','ffdeftext','ffentrymcr','ffexitmcr','ffformat','ffhelptext','ffl',
      'ffname','ffstattext','field','file','filetbl','fldinst','fldrslt','fldtype',
      'fname','fontemb','fontfile','fonttbl','footer','footerf','footerl','footerr',
      'footnote','formfield','ftncn','ftnsep','ftnsepc','g','generator','gridtbl',
      'header','headerf','headerl','headerr','hl','hlfr','hlinkbase','hlloc','hlsrc',
      'hsv','htmltag','info','keycode','keywords','latentstyles','lchars','levelnumbers',
      'leveltext','lfolevel','linkval','list','listlevel','listname','listoverride',
      'listoverridetable','listpicture','liststylename','listtable','listtext',
      'lsdlockedexcept','macc','maccPr','mailmerge','maln','malnScr','manager','margPr',
      'mbar','mbarPr','mbaseJc','mbegChr','mborderBox','mborderBoxPr','mbox','mboxPr',
      'mchr','mcount','mctrlPr','md','mdeg','mdegHide','mden','mdiff','mdPr','me',
      'mendChr','meqArr','meqArrPr','mf','mfName','mfPr','mfunc','mfuncPr','mgroupChr',
      'mgroupChrPr','mgrow','mhideBot','mhideLeft','mhideRight','mhideTop','mhtmltag',
      'mlim','mlimloc','mlimlow','mlimlowPr','mlimupp','mlimuppPr','mm','mmaddfieldname',
      'mmath','mmathPict','mmathPr','mmaxdist','mmc','mmcJc','mmconnectstr',
      'mmconnectstrdata','mmcPr','mmcs','mmdatasource','mmheadersource','mmmailsubject',
      'mmodso','mmodsofilter','mmodsofldmpdata','mmodsomappedname','mmodsoname',
      'mmodsorecipdata','mmodsosort','mmodsosrc','mmodsotable','mmodsoudl',
      'mmodsoudldata','mmodsouniquetag','mmPr','mmquery','mmr','mnary','mnaryPr',
      'mnoBreak','mnum','mobjDist','moMath','moMathPara','moMathParaPr','mopEmu',
      'mphant','mphantPr','mplcHide','mpos','mr','mrad','mradPr','mrPr','msepChr',
      'mshow','mshp','msPre','msPrePr','msSub','msSubPr','msSubSup','msSubSupPr','msSup',
      'msSupPr','mstrikeBLTR','mstrikeH','mstrikeTLBR','mstrikeV','msub','msubHide',
      'msup','msupHide','mtransp','mtype','mvertJc','mvfmf','mvfml','mvtof','mvtol',
      'mzeroAsc','mzeroDesc','mzeroWid','nesttableprops','nextfile','nonesttables',
      'objalias','objclass','objdata','object','objname','objsect','objtime','oldcprops',
      'oldpprops','oldsprops','oldtprops','oleclsid','operator','panose','password',
      'passwordhash','pgp','pgptbl','picprop','pict','pn','pnseclvl','pntext','pntxta',
      'pntxtb','printim','private','propname','protend','protstart','protusertbl','pxe',
      'result','revtbl','revtim','rsidtbl','rxe','shp','shpgrp','shpinst',
      'shppict','shprslt','shptxt','sn','sp','staticval','stylesheet','subject','sv',
      'svb','tc','template','themedata','title','txe','ud','upr','userprops',
      'wgrffmtfilter','windowcaption','writereservation','writereservhash','xe','xform',
      'xmlattrname','xmlattrvalue','xmlclose','xmlname','xmlnstbl',
      'xmlopen',
    ))
    # Translation of some special characters.
    specialchars = {
      'par': '\n',
      'sect': '\n\n',
      'page': '\n\n',
      'line': '\n',
      'tab': '\t',
      'emdash': '\u2014',
      'endash': '\u2013',
      'emspace': '\u2003',
      'enspace': '\u2002',
      'qmspace': '\u2005',
      'bullet': '\u2022',
      'lquote': '\u2018',
      'rquote': '\u2019',
      'ldblquote': '\201C',
      'rdblquote': '\u201D',
    }
    stack = []
    ignorable = False       # Whether this group (and all inside it) are "ignorable".
    ucskip = 1              # Number of ASCII characters to skip after a unicode character.
    curskip = 0             # Number of ASCII characters left to skip
    out = []                # Output buffer.
    for match in pattern.finditer(text):
      word,arg,hex,char,brace,tchar = match.groups()
      if brace:
         curskip = 0
         if brace == '{':
            # Push state
            stack.append((ucskip,ignorable))
         elif brace == '}':
            # Pop state
            ucskip,ignorable = stack.pop()
      elif char: # \x (not a letter)
         curskip = 0
         if char == '~':
            if not ignorable:
                out.append('\xA0')
         elif char in '{}\\':
            if not ignorable:
               out.append(char)
         elif char == '*':
            ignorable = True
      elif word: # \foo
         curskip = 0
         if word in destinations:
            ignorable = True
         elif ignorable:
            pass
         elif word in specialchars:
            out.append(specialchars[word])
         elif word == 'uc':
            ucskip = int(arg)
         elif word == 'u':
            c = int(arg)
            if c < 0: c += 0x10000
            if c > 127: out.append(chr(c)) #NOQA
            else: out.append(chr(c))
            curskip = ucskip
      elif hex: # \'xx
         if curskip > 0:
            curskip -= 1
         elif not ignorable:
            c = int(hex,16)
            if c > 127: out.append(chr(c)) #NOQA
            else: out.append(chr(c))
      elif tchar:
         if curskip > 0:
            curskip -= 1
         elif not ignorable:
            out.append(tchar)
    return ''.join(out)


if __name__ == '__main__':
    import doctest
    # noinspection PyUnresolvedReferences
    from pyqae import nlp

    doctest.testmod(nlp, verbose=True, optionflags=doctest.ELLIPSIS)
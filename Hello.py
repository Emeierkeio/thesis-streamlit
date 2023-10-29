# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
stop_words_txt = "a,abbastanza,abbia,abbiamo,abbiano,abbiate,accidenti,ad,adesso,affinché,agl,agli,ahime,ahimè,ai,al,alcuna,alcuni,alcuno,all,alla,alle,allo,allora,altre,altri,altrimenti,altro,altrove,altrui,anche,ancora,anni,anno,ansa,anticipo,assai,attesa,attraverso,avanti,avemmo,avendo,avente,aver,avere,averlo,avesse,avessero,avessi,avessimo,aveste,avesti,avete,aveva,avevamo,avevano,avevate,avevi,avevo,avrai,avranno,avrebbe,avrebbero,avrei,avremmo,avremo,avreste,avresti,avrete,avrà,avrò,avuta,avute,avuti,avuto,basta,ben,bene,benissimo,brava,bravo,buono,c,caso,cento,certa,certe,certi,certo,che,chi,chicchessia,chiunque,ci,ciascuna,ciascuno,cima,cinque,cio,cioe,cioè,circa,citta,città,ciò,co,codesta,codesti,codesto,cogli,coi,col,colei,coll,coloro,colui,come,cominci,comprare,comunque,con,concernente,conclusione,consecutivi,consecutivo,consiglio,contro,cortesia,cos,cosa,cosi,così,cui,d,da,dagl,dagli,dai,dal,dall,dalla,dalle,dallo,dappertutto,davanti,degl,degli,dei,del,dell,della,delle,dello,dentro,detto,deve,devo,di,dice,dietro,dire,dirimpetto,diventa,diventare,diventato,dopo,doppio,dov,dove,dovra,dovrà,dovunque,due,dunque,durante,e,ebbe,ebbero,ebbi,ecc,ecco,ed,effettivamente,egli,ella,entrambi,eppure,era,erano,eravamo,eravate,eri,ero,esempio,esse,essendo,esser,essere,essi,ex,fa,faccia,facciamo,facciano,facciate,faccio,facemmo,facendo,facesse,facessero,facessi,facessimo,faceste,facesti,faceva,facevamo,facevano,facevate,facevi,facevo,fai,fanno,farai,faranno,fare,farebbe,farebbero,farei,faremmo,faremo,fareste,faresti,farete,farà,farò,fatto,favore,fece,fecero,feci,fin,finalmente,finche,fine,fino,forse,forza,fosse,fossero,fossi,fossimo,foste,fosti,fra,frattempo,fu,fui,fummo,fuori,furono,futuro,generale,gente,gia,giacche,giorni,giorno,giu,già,gli,gliela,gliele,glieli,glielo,gliene,grande,grazie,gruppo,ha,haha,hai,hanno,ho,i,ie,ieri,il,improvviso,in,inc,indietro,infatti,inoltre,insieme,intanto,intorno,invece,io,l,la,lasciato,lato,le,lei,li,lo,lontano,loro,lui,lungo,luogo,là,ma,macche,magari,maggior,mai,male,malgrado,malissimo,me,medesimo,mediante,meglio,meno,mentre,mesi,mezzo,mi,mia,mie,miei,mila,miliardi,milioni,minimi,mio,modo,molta,molti,moltissimo,molto,momento,mondo,ne,negl,negli,nei,nel,nell,nella,nelle,nello,nemmeno,neppure,nessun,nessuna,nessuno,niente,no,noi,nome,non,nondimeno,nonostante,nonsia,nostra,nostre,nostri,nostro,novanta,nove,nulla,nuovi,nuovo,o,od,oggi,ogni,ognuna,ognuno,oltre,oppure,ora,ore,osi,ossia,ottanta,otto,paese,parecchi,parecchie,parecchio,parte,partendo,peccato,peggio,per,perche,perchè,perché,percio,perciò,perfino,pero,persino,persone,però,piedi,pieno,piglia,piu,piuttosto,più,po,pochissimo,poco,poi,poiche,possa,possedere,posteriore,posto,potrebbe,preferibilmente,presa,press,prima,primo,principalmente,probabilmente,promesso,proprio,puo,pure,purtroppo,può,qua,qualche,qualcosa,qualcuna,qualcuno,quale,quali,qualunque,quando,quanta,quante,quanti,quanto,quantunque,quarto,quasi,quattro,quel,quella,quelle,quelli,quello,quest,questa,queste,questi,questo,qui,quindi,quinto,realmente,recente,recentemente,registrazione,relativo,riecco,rispetto,salvo,sara,sarai,saranno,sarebbe,sarebbero,sarei,saremmo,saremo,sareste,saresti,sarete,sarà,sarò,scola,scopo,scorso,se,secondo,seguente,seguito,sei,sembra,sembrare,sembrato,sembrava,sembri,sempre,senza,sette,si,sia,siamo,siano,siate,siete,sig,solito,solo,soltanto,sono,sopra,soprattutto,sotto,spesso,sta,stai,stando,stanno,starai,staranno,starebbe,starebbero,starei,staremmo,staremo,stareste,staresti,starete,starà,starò,stata,state,stati,stato,stava,stavamo,stavano,stavate,stavi,stavo,stemmo,stessa,stesse,stessero,stessi,stessimo,stesso,steste,stesti,stette,stettero,stetti,stia,stiamo,stiano,stiate,sto,su,sua,subito,successivamente,successivo,sue,sugl,sugli,sui,sul,sull,sulla,sulle,sullo,suo,suoi,tale,tali,talvolta,tanto,te,tempo,terzo,th,ti,titolo,tra,tranne,tre,trenta,triplo,troppo,trovato,tu,tua,tue,tuo,tuoi,tutta,tuttavia,tutte,tutti,tutto,uguali,ulteriore,ultimo,un,una,uno,uomo,va,vai,vale,vari,varia,varie,vario,verso,vi,vicino,visto,vita,voi,volta,volte,vostra,vostre,vostri,vostro,è"

stop_word = stop_words_txt.split(',')

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("DataFrame XIX Legislature")
    
    df = get_data()

    st.dataframe(filter_dataframe(df))

def get_data():
  url = 'https://drive.google.com/file/d/1RttXvwXT21mKsp0fqb9N1fOvNcqnUbZ7/view?usp=sharing'
  path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]


  return pd.read_csv(path, lineterminator='\n')


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


if __name__ == "__main__":
    run()

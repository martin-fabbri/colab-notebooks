import streamlit as st
import package

A = 1
B = 1

st.title("My first cool app")
st.write(f"{A} + {B} = {package.add(A, B)}")
import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
from scipy.signal import butter,filtfilt
import plotly.graph_objects as go

st.header('Communication')

communication_type = st.radio(
    "Choose Coomnuication type",
    ["Analog Communication","Digital Communication"],
    index=0,
)

#Message Signal -------------------------------------------------------

t = np.linspace(0, 1, 1000)

if communication_type == "Digital Communication":

    amp = st.slider('Message Amplitude', 0, 10, 3)

    times = 1000
    # bits = [1,0,1,0,1,0,0,1,0,1]
    binary_input = st.text_input("Enter a binary string (0 and 1):",'1011101')
    if set(binary_input) <= {'0', '1'}:
        st.write(f"You entered: {binary_input}")
        bits = []
        for b in binary_input:
            bits.append(int(b))
    else:
        st.warning("Please enter a valid binary string containing only 0s and 1s.")

    times = int(len(bits)*(times//len(bits)))

    t = np.linspace(0, 1, times)

    st.write(times)

    # amp = 3
    B = []
    for b in bits:
        if b == 1:
            B.append(int(amp))
        else:
            B.append(0)
    bits = B



    time_per_bit = times//len(bits)
    digital_data = []
    for i in bits:
        digital_data += [i]*time_per_bit

    modulator = digital_data


elif communication_type == "Analog Communication":
    
    st.subheader('Message Signal')  
    message_type = st.radio(
        "Choose Message Signal",
        ["Sine wave","Cos wave"],
        index=0,
    )

    A_m = st.slider('Message Amplitude', 0, 50, 2)
    f_m = st.slider('Message frquency', 0, 50, 2)

    if message_type == "Sine wave":
        modulator = A_m*np.sin(2*np.pi*f_m*t)
        st.subheader(f':blue[Message : y = {A_m}sin(2π{f_m}t)]')
    elif message_type == "Cos wave":
        modulator = A_m*np.cos(2*np.pi*f_m*t)
        st.subheader(f':blue[Message : y = {A_m}cos(2π{f_m}t)]')

st.line_chart(modulator)

#Carrier Signal -------------------------------------------------------

st.subheader('Carrier Signal')
carrier_idx = st.radio(
    "Choose Carrier Signal",
    ["Sine wave","Cos wave"],
    index=0,
)

A_c = st.slider('Carrier Amplitude', 0, 50, 10)
f_c = st.slider('Carrier Frquency', 0, 30, 10)

if carrier_idx == 'Sine wave':
    carrier = A_c*np.sin(2*np.pi*f_c*t)
    st.subheader(f':blue[Carrier : y = {A_c}sin(2π{f_c}t)]')
elif carrier_idx == 'Cos wave':
    carrier = A_c*np.cos(2*np.pi*f_c*t)
    st.subheader(f':blue[Carrier : y = {A_c}cos(2π{f_c}t)]')

st.line_chart(carrier)


#Modulation -------------------------------------------------------

st.subheader('Modulation')
product = np.array(modulator)*np.array(carrier)

data = {
    '1_modulator': modulator,
    '2_carrier': carrier,
    '3_modulated': product
}

df = pd.DataFrame(data)

st.line_chart(df)

#Adding Noise -------------------------------------------------------

st.subheader('Adding Noise')
target_snr_db = st.slider('SNR(db)', 0, 100, 20)



x_volts = product
x_watts = x_volts ** 2
x_db = 10 * np.log10(x_watts)

# Calculate signal power and convert to dB 
sig_avg_watts = np.mean(x_watts)
sig_avg_db = 10 * np.log10(sig_avg_watts)
# Calculate noise according to [2] then convert to watts
noise_avg_db = sig_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
# Generate an sample of white noise
mean_noise = 0
noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
# Noise up the original signal
y_volts = x_volts + noise_volts

data = {
    'y_volts': y_volts,
}
df = pd.DataFrame(data)
st.line_chart(df)


#Filter Noise -------------------------------------------------------
st.subheader('Filter Noise')
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
# Filter requirements.
T = 5.0         # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
T = st.slider('Sample Period', 0, 20, 5)
fs = st.slider('sample rate, Hz', 0, 100, 30)
cutoff = st.slider('cutoff frequency', 0, 10, 2)

nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples


data = y_volts

y = butter_lowpass_filter(data, cutoff, fs, order)
fig = go.Figure()
fig.add_trace(go.Scatter(
            y = data,
            line =  dict(shape =  'spline' ),
            name = 'signal with noise',
            
            ))
fig.add_trace(go.Scatter(
            y = y,
            line =  dict(shape =  'spline' ),
            name = 'filtered signal'
            ))
# fig.show()
st.plotly_chart(fig, use_container_width=True)









# Adding noise using target SNR
# Set a target SNR
# target_snr_db = 10


# st.write("Carrier Amplitude : ", A_c)
# st.write("Carrier Frquency : ", f_c)





# st.title
# st.header
# st.subheader


# carrier = A_c*np.sin(2*np.pi*f_c*t)
# modulator = A_m*np.sin(2*np.pi*f_m*t) + A_m*np.sin(4*np.pi*f_m*t)
# modulator = A_m*np.sin(2*np.pi*f_m*t)

# product = A_c*(1+modulation_index*np.cos(2*np.pi*f_m*t))*np.cos(2*np.pi*f_c*t)



# A_c = 10.0 #carrier amplitude
# f_c = 30.0  #carrier frquency
# A_m = 2.0 #message amplitude
# f_m = 2.0 #message frquency
# modulation_index = 2

# x = np.array([carrier,modulator,product])

# st.write(x.shape)

# chart_data = pd.DataFrame(np.array(x), columns=["a", "b", "c"])
# modulator_data = pd.DataFrame(modulator, columns=["modulator"])
# chart_data = pd.DataFrame(carrier, columns=["carrier"])
# modulated_data = pd.DataFrame(product, columns=["modulated"])

# column_order = ['1.modulator', '2.carrier', '3.modulated']
# df = df[column_order]

# st.write(df)

# st.line_chart(modulator_data)
# st.line_chart(chart_data)
# st.line_chart(modulated_data)



# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])








# import numpy as np
# import matplotlib.pyplot as plt

# #Carrier wave c(t)=A_c*cos(2*pi*f_c*t)
# #Modulating wave m(t)=A_m*cos(2*pi*f_m*t)
# #Modulated wave s(t)=A_c[1+mu*cos(2*pi*f_m*t)]cos(2*pi*f_c*t)

# # A_c = float(input('Enter carrier amplitude: '))
# # f_c = float(input('Enter carrier frquency: '))
# # A_m = float(input('Enter message amplitude: '))
# # f_m = float(input('Enter message frquency: '))

# A_c = 10.0 #carrier amplitude
# f_c = 30.0  #carrier frquency
# A_m = 2.0 #message amplitude
# f_m = 2.0 #message frquency
# modulation_index = 2
# # modulation_index = float(input('Enter modulation index: '))

# t = np.linspace(0, 1, 1000)

# carrier = A_c*np.cos(2*np.pi*f_c*t)
# modulator = A_m*np.cos(2*np.pi*f_m*t) + A_m*np.cos(4*np.pi*f_m*t)
# # product = A_c*(1+modulation_index*np.cos(2*np.pi*f_m*t))*np.cos(2*np.pi*f_c*t)
# product = np.array(modulator)*np.array(carrier)

# plt.subplot(3,1,1)
# plt.title('Amplitude Modulation')
# plt.plot(modulator,'g')
# plt.ylabel('Amplitude')
# plt.xlabel('Message signal')

# plt.subplot(3,1,2)
# plt.plot(carrier, 'r')
# plt.ylabel('Amplitude')
# plt.xlabel('Carrier signal')

# plt.subplot(3,1,3)
# plt.plot(product, color="purple")
# plt.ylabel('Amplitude')
# plt.xlabel('AM signal')

# plt.subplots_adjust(hspace=1)
# plt.rc('font', size=15)
# fig = plt.gcf()
# fig.set_size_inches(16, 9)

# fig.savefig('Amplitude Modulation.png', dpi=100)





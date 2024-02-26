from flask import Flask, render_template, redirect, request
from joblib import load
import pandas as pd

xgb_model = load('xgb_model.joblib')
log_reg_model = load('win_predictor_model.joblib')

#nn_model = load_model('neural_net_model.h5')

data_cols = ['current_score', 'wickets_remaining', 'balls_left', 'last_five', 'batting_team_Afghanistan', 'batting_team_Australia', 'batting_team_England', 'batting_team_India', 'batting_team_Kenya', 'batting_team_Netherlands', 'batting_team_New Zealand', 'batting_team_Pakistan', 'batting_team_Papua New Guinea', 'batting_team_South Africa', 'batting_team_United Arab Emirates', 'batting_team_West Indies', 'batting_team_Zimbabwe', 'bowling_team_Afghanistan', 'bowling_team_Australia', 'bowling_team_England', 'bowling_team_India', 'bowling_team_Kenya', 'bowling_team_Netherlands', 'bowling_team_New Zealand', 'bowling_team_Pakistan', 'bowling_team_Papua New Guinea', 'bowling_team_South Africa', 'bowling_team_United Arab Emirates', 'bowling_team_West Indies', 'bowling_team_Zimbabwe', 'city_Abu Dhabi', 'city_Adelaide', 'city_Ahmedabad', 'city_Amstelveen', 'city_Antigua', 'city_Auckland', 'city_Bangalore', 'city_Barbados', 'city_Basseterre', 'city_Bengaluru', 'city_Birmingham', 'city_Bloemfontein', 'city_Brisbane', 'city_Bristol', 'city_Bulawayo', 'city_Canberra', 'city_Cape Town', 'city_Cardiff', 'city_Carrara', 'city_Centurion', 'city_Chandigarh', 'city_Chattogram', 'city_Chennai', 'city_Chester-le-Street', 'city_Chittagong', 'city_Christchurch', 'city_Colombo', 'city_Cuttack', 'city_Delhi', 'city_Dhaka', 'city_Dharamsala', 'city_Dharmasala', 'city_Dominica', 'city_Dubai', 'city_Dublin', 'city_Durban', 'city_East London', 'city_Edinburgh', 'city_Fatullah', 'city_Gros Islet', 'city_Guwahati', 'city_Guyana', 'city_Hambantota', 'city_Hamilton', 'city_Harare', 'city_Hobart', 'city_Hyderabad', 'city_Jamaica', 'city_Johannesburg', 'city_Kanpur', 'city_Karachi', 'city_Kimberley', 'city_Kolkata', 'city_Lahore', 'city_Lauderhill', 'city_London', 'city_Lucknow', 'city_Manchester', 'city_Melbourne', 'city_Mirpur', 'city_Mount Maunganui', 'city_Mumbai', 'city_Nagpur', 'city_Nairobi', 'city_Napier', 'city_Nelson', 'city_Nottingham', 'city_Paarl', 'city_Pallekele', 'city_Perth', 'city_Port Elizabeth', 'city_Potchefstroom', 'city_Providence', 'city_Pune', 'city_Rajkot', 'city_Ranchi', 'city_Rawalpindi', 'city_Rotterdam', 'city_Sharjah', 'city_Southampton', 'city_St Kitts', 'city_St Lucia', 'city_St Vincent', 'city_Sydney', 'city_Sylhet', 'city_Taunton', 'city_The Hague', 'city_Thiruvananthapuram', 'city_Trinidad', 'city_Visakhapatnam', 'city_Wellington', 'city_Windhoek']
data_cols1 = ['runs_left', 'balls_left', 'wickets_remaining', 'target', 'crr', 'rrr', 'batting_team_Afghanistan', 'batting_team_Australia', 'batting_team_England', 'batting_team_India', 'batting_team_Kenya', 'batting_team_Netherlands', 'batting_team_New Zealand', 'batting_team_Pakistan', 'batting_team_Papua New Guinea', 'batting_team_South Africa', 'batting_team_United Arab Emirates', 'batting_team_West Indies', 'batting_team_Zimbabwe', 'bowling_team_Afghanistan', 'bowling_team_Australia', 'bowling_team_England', 'bowling_team_India', 'bowling_team_Kenya', 'bowling_team_Netherlands', 'bowling_team_New Zealand', 'bowling_team_Pakistan', 'bowling_team_Papua New Guinea', 'bowling_team_South Africa', 'bowling_team_United Arab Emirates', 'bowling_team_West Indies', 'bowling_team_Zimbabwe', 'city_Abu Dhabi', 'city_Adelaide', 'city_Ahmedabad', 'city_Amstelveen', 'city_Antigua', 'city_Auckland', 'city_Bangalore', 'city_Barbados', 'city_Basseterre', 'city_Bengaluru', 'city_Birmingham', 'city_Bloemfontein', 'city_Brisbane', 'city_Bristol', 'city_Bulawayo', 'city_Canberra', 'city_Cape Town', 'city_Cardiff', 'city_Carrara', 'city_Centurion', 'city_Chandigarh', 'city_Chattogram', 'city_Chennai', 'city_Chester-le-Street', 'city_Chittagong', 'city_Christchurch', 'city_Colombo', 'city_Cuttack', 'city_Delhi', 'city_Dhaka', 'city_Dharamsala', 'city_Dharmasala', 'city_Dominica', 'city_Dubai', 'city_Dublin', 'city_Durban', 'city_East London', 'city_Edinburgh', 'city_Fatullah', 'city_Gros Islet', 'city_Guwahati', 'city_Guyana', 'city_Hambantota', 'city_Hamilton', 'city_Harare', 'city_Hobart', 'city_Hyderabad', 'city_Jamaica', 'city_Johannesburg', 'city_Kanpur', 'city_Karachi', 'city_Kimberley', 'city_Kolkata', 'city_Lahore', 'city_Lauderhill', 'city_London', 'city_Lucknow', 'city_Manchester', 'city_Melbourne', 'city_Mirpur', 'city_Mount Maunganui', 'city_Mumbai', 'city_Nagpur', 'city_Nairobi', 'city_Napier', 'city_Nelson', 'city_Nottingham', 'city_Paarl', 'city_Pallekele', 'city_Perth', 'city_Port Elizabeth', 'city_Potchefstroom', 'city_Providence', 'city_Pune', 'city_Rajkot', 'city_Ranchi', 'city_Rawalpindi', 'city_Rotterdam', 'city_Sharjah', 'city_Southampton', 'city_St Kitts', 'city_St Lucia', 'city_St Vincent', 'city_Sydney', 'city_Sylhet', 'city_Taunton', 'city_The Hague', 'city_Thiruvananthapuram', 'city_Trinidad', 'city_Visakhapatnam', 'city_Wellington', 'city_Windhoek']

teams = ['India', 'West Indies', 'Australia', 'England', 'South Africa', 'Afghanistan',
 'Netherlands', 'Papua New Guinea', 'New Zealand', 'United Arab Emirates',
 'Pakistan', 'Zimbabwe', 'Kenya']

cities = ['Lauderhill', 'St Lucia', 'Bangalore', 'Nottingham', 'Cape Town', 'Dubai',
 'Wellington', 'Harare', 'Durban', 'Hamilton', 'Chandigarh', 'Sharjah',
 'Colombo', 'Southampton', 'Melbourne', 'Rawalpindi', 'Trinidad', 'Auckland',
 'Abu Dhabi', 'Johannesburg', 'Sylhet', 'Bristol', 'London', 'Kanpur', 'Sydney',
 'Rotterdam', 'Chittagong', 'Nairobi', 'Mirpur', 'Potchefstroom', 'Windhoek',
 'Hyderabad', 'Barbados', 'Mumbai', 'Cardiff', 'Christchurch', 'St Vincent',
 'Chennai', 'Manchester', 'Ranchi', 'Thiruvananthapuram', 'Adelaide',
 'Bloemfontein', 'Delhi', 'Rajkot', 'Dublin', 'Hobart', 'Chester-le-Street',
 'St Kitts', 'Napier', 'Port Elizabeth', 'Fatullah', 'Amstelveen', 'Bengaluru',
 'Gros Islet', 'Centurion', 'Kolkata', 'Bulawayo', 'Pallekele', 'Cuttack',
 'Birmingham', 'Lucknow', 'Perth', 'Nagpur', 'Karachi', 'Dhaka', 'Antigua',
 'Mount Maunganui', 'Jamaica', 'Carrara', 'Edinburgh', 'Brisbane',
 'East London', 'Canberra', 'Paarl', 'Pune', 'Chattogram', 'Basseterre',
 'Dharmasala', 'Dominica', 'Lahore', 'Ahmedabad', 'Hambantota', 'Providence',
 'Nelson', 'Guyana', 'Guwahati', 'Kimberley', 'Visakhapatnam', 'The Hague',
 'Taunton', 'Dharamsala']


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_score', methods=['GET','POST'])
def predict_score():
    prediction = 0
    form_data = None
    if request.method == 'POST':
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        city = request.form.get('city')
        runs_scored = int(request.form.get('runs'))
        overs_completed = int(request.form.get('overs'))
        wickets_lost = int(request.form.get('wickets'))
        last_five_overs = int(request.form.get('last_five'))

        form_data = {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_scored': runs_scored,
            'overs_completed': overs_completed,
            'wickets_lost': wickets_lost,
            'last_five_overs': last_five_overs
        }

        # preparing the data to fit the model
        score_df = prepare_input_predict_score(batting_team, bowling_team, city, runs_scored, overs_completed, wickets_lost, last_five_overs)
        score_df.to_csv('score_df.csv', index=False)
        
        # model prediction
        prediction = xgb_model.predict(score_df)[0]
        print(prediction)

    return render_template('predict_score.html', prediction=prediction, form_data=form_data)

def prepare_input_predict_score(batting_team, bowling_team, city, runs_scored, overs_completed, wickets_lost, last_five_overs):
    input_data = {col: 0 for col in data_cols}

    input_data['current_score'] = runs_scored
    input_data['wickets_remaining'] = 10 - wickets_lost
    input_data['balls_left'] = (20 - overs_completed) * 6
    input_data['last_five'] = last_five_overs


    at_bat = "batting_team_" + batting_team
    input_data[at_bat] = 1

    at_bowl = "bowling_team_" + bowling_team
    input_data[at_bowl] = 1

    at_city = "city_" + city
    input_data[at_city] = 1

    input_df = pd.DataFrame([input_data], columns=data_cols)

    return input_df

@app.route('/win_prob', methods=['GET','POST'])
def predict_win_prob():
    probability = None
    batting_team = None
    bowling_team = None
    form_data = None
    if request.method == 'POST':
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        city = request.form.get('city')
        runs_scored = int(request.form.get('runs'))
        overs_completed = int(request.form.get('overs'))
        wickets_lost = int(request.form.get('wickets'))
        target_score = int(request.form.get('target_score'))

        form_data = {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_scored': runs_scored,
            'overs_completed': overs_completed,
            'wickets_lost': wickets_lost,
            'target_score': target_score
        }

        # preparing the data to fit the model
        win_prob_df = prepare_input_win_prob(batting_team, bowling_team, city, runs_scored, overs_completed, wickets_lost, target_score)
        win_prob_df.to_csv('win_prob_df.csv', index=False)
        
        # win probability prediction
        probability = round(log_reg_model.predict_proba(win_prob_df)[0][1],2)

    return render_template('win_prob.html', prediction=probability, batting_team=batting_team, bowling_team=bowling_team, form_data=form_data)


def prepare_input_win_prob(batting_team, bowling_team, city, runs_scored, overs_completed, wickets_lost, target_score):
    input_data = {col: 0 for col in data_cols1}

    input_data['current_score'] = runs_scored
    input_data['runs_left'] = target_score - runs_scored
    input_data['balls_left'] = (20 - overs_completed) * 6
    input_data['wickets_remaining'] = 10 - wickets_lost
    input_data['target_score'] = target_score
    input_data['crr'] = runs_scored / ((120 - input_data['balls_left']) / 6)
    input_data['rrr'] = (target_score - runs_scored) / (input_data['balls_left'] / 6) 


    at_bat = "batting_team_" + batting_team
    input_data[at_bat] = 1

    at_bowl = "bowling_team_" + bowling_team
    input_data[at_bowl] = 1

    at_city = "city_" + city
    input_data[at_city] = 1

    input_df = pd.DataFrame([input_data], columns=data_cols1)

    return input_df

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='localhost',port=8000, debug=True)

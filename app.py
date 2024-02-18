from flask import Flask, render_template, redirect, request

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    prediction = 0
    if request.method == 'POST':
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        city = request.form.get('city')
        runs_scored = int(request.form.get('runs'))
        overs_completed = int(request.form.get('overs'))
        wickets_lost = int(request.form.get('wickets'))
        last_five_overs = int(request.form.get('last_five'))


        # replace with model
        prediction = runs_scored + overs_completed 
    
    
    return render_template('index.html', prediction=prediction)
        


if __name__ == '__main__':
    app.run(debug=True)
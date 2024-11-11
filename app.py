from flask import Flask, render_template, request, url_for, session, redirect, flash
from flask_session import Session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'session:'
Session(app)

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.scatter(X, Y, color='blue')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red')
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.clf()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.hist(slopes, bins=30, alpha=0.5, label='Slopes')
    plt.hist(intercepts, bins=30, alpha=0.5, label='Intercepts')
    plt.legend(loc='upper right')
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.clf()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Check if session variables are set
    if not all(k in session for k in ("N", "S", "slope", "intercept", "slopes", "intercepts", "beta0", "beta1")):
        flash("Please generate data first.")
        return redirect(url_for("index"))

    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        p_value = np.mean(np.abs(simulated_stats) >= np.abs(observed_stat))

    # If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    fun_message = None
    if p_value <= 0.0001:
        fun_message = "Wow! You've encountered a rare event!"

    # Plot histogram of simulated statistics
    plt.hist(simulated_stats, bins=30, alpha=0.5, label='Simulated Stats')
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Stat')
    plt.axvline(hypothesized_value, color='blue', linestyle='dashed', linewidth=2, label='Hypothesized Value')
    plt.legend(loc='upper right')
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.clf()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        p_value=p_value,
        fun_message=fun_message,
        N=N,
        mu=session.get("mu"),
        sigma2=session.get("sigma2"),
        beta0=beta0,
        beta1=beta1,
        S=S
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Check if session variables are set
    if not all(k in session for k in ("N", "mu", "sigma2", "beta0", "beta1", "S", "X", "Y", "slope", "intercept", "slopes", "intercepts")):
        flash("Please generate data first.")
        return redirect(url_for("index"))

    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)

    # Calculate confidence interval for the parameter estimate
    ci_lower = mean_estimate - 1.96 * std_estimate / np.sqrt(S)
    ci_upper = mean_estimate + 1.96 * std_estimate / np.sqrt(S)

    # Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot the individual estimates and confidence interval
    plt.scatter(range(S), estimates, color='gray', alpha=0.5)
    plt.axhline(mean_estimate, color='red', linestyle='dashed', linewidth=2, label='Mean Estimate')
    plt.axhline(ci_lower, color='blue', linestyle='dashed', linewidth=2, label='CI Lower')
    plt.axhline(ci_upper, color='blue', linestyle='dashed', linewidth=2, label='CI Upper')
    plt.axhline(true_param, color='green', linestyle='dashed', linewidth=2, label='True Parameter')
    plt.legend(loc='upper right')
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.clf()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S
    )

if __name__ == "__main__":
    app.run(debug=True)
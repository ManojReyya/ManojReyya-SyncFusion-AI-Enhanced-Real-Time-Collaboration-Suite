from flask import Flask, render_template, send_file, request, send_from_directory, jsonify, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, TextAreaField
from wtforms.validators import DataRequired, Length
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_user, LoginManager, login_required, current_user, logout_user,UserMixin
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import google.generativeai as genai
from functools import wraps
from flask_bcrypt import Bcrypt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import os
import json
import requests
import re
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Initialize Flask App
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["SECRET_KEY"] = "my-secrets"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///SyncFusion.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Database & Login Manager
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# Flask-Mail Configuration
# 🔑 Secret Key for Flash Messages
app.secret_key = "super_secret_key_123"

app.config["MAIL_SERVER"] = "smtp.gmail.com"  # SMTP Server
app.config["MAIL_PORT"] = 587  # Port for TLS
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = "syncfusion.collab@gmail.com"  # Your email
app.config["MAIL_PASSWORD"] = "nvkb umwm kjpq nyjc"  # App password or actual password
app.config["MAIL_DEFAULT_SENDER"] = "syncfusioncollab@gmail.com"

mail = Mail(app)
bcrypt = Bcrypt(app)

#Gemini API Model
genai.configure(api_key="AIzaSyAC0LZmrIibBm88z5inZnF1CZZbVA-RCDw")

# Configure the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

# Database Models

# Ensure upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# User Authentication Model
class Register(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(50), unique=True, nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(10), default="user")  # Default role is "user" 
    # Property to check if user is admin
    @property
    def admin(self):
        return self.role == "admin"

    @property
    def user(self):
        return self.role == "user"

    def is_active(self):
        return True

    def get_id(self):
        return str(self.id)

    def is_authenticated(self):
        return True


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('register.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow, nullable=False)
    status = db.Column(db.String(20), default="Present", nullable=False)

    # Relationship with Register model
    user = db.relationship('Register', backref=db.backref('attendance', lazy=True))

class Poll(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(255), nullable=False)

class Option(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    poll_id = db.Column(db.Integer, db.ForeignKey('poll.id'), nullable=False)
    text = db.Column(db.String(255), nullable=False)
    votes = db.Column(db.Integer, default=0)

class Vote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    poll_id = db.Column(db.Integer, db.ForeignKey('poll.id'), nullable=False)
    user_id = db.Column(db.String(255), nullable=False)  # Tracking votes per user

# Define Task Record Model
class TaskRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    task = db.Column(db.String(255), nullable=False)
    duration = db.Column(db.String(50), nullable=False)
    
# Define Project Model
class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    task = db.Column(db.Text, nullable=False)
    start_date = db.Column(db.String(50), nullable=False)
    deadline = db.Column(db.String(50), nullable=False)
    priority = db.Column(db.Integer, default=1)
    status = db.Column(db.String(50), default="To Do")

@login_manager.user_loader
def load_user(user_id):
    return Register.query.get(int(user_id))

# Create DB Tables
with app.app_context():
    db.create_all()

# Registration Form
class RegistrationForm(FlaskForm):
    email = EmailField(label='Email', validators=[DataRequired()])
    first_name = StringField(label="First Name", validators=[DataRequired()])
    last_name = StringField(label="Last Name", validators=[DataRequired()])
    username = StringField(label="Username", validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField(label="Password", validators=[DataRequired(), Length(min=8, max=20)])

# Login Form
class LoginForm(FlaskForm):
    email = EmailField(label='Email', validators=[DataRequired()])
    password = PasswordField(label="Password", validators=[DataRequired()])


# User Authentication Routes
@app.route("/")
def home():
    return redirect(url_for("login"))
@app.route("/login", methods=["POST", "GET"])
def login():
    form = LoginForm()
    if request.method == "POST" and form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        user = Register.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):  # Secure comparison
            login_user(user)

            # Check if attendance already exists for today
            today = datetime.utcnow().date()
            existing_attendance = Attendance.query.filter_by(user_id=user.id, date=today).first()

            if not existing_attendance:
                # Insert a new attendance record
                new_attendance = Attendance(user_id=user.id, date=today, status="Present")
                db.session.add(new_attendance)
                db.session.commit()

            return redirect(url_for("dashboard"))

    return render_template("login.html", form=form)
@app.route("/logout", methods=["GET"])
@login_required  
def logout():
    logout_user()
    flash("You have been logged out successfully!", "info")
    return redirect(url_for("login"))
@app.route("/register", methods=["POST", "GET"])
def register():
    form = RegistrationForm()
    if request.method == "POST" and form.validate_on_submit():
        first_user = Register.query.first()  # Check if any user exists
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        new_user = Register(
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            username=form.username.data,
            password=hashed_password,
            role="admin" if first_user is None else "user"  # First user is admin
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Account created successfully! You can now log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", form=form)


# After login redirect to Dashboard
@app.route("/dashboard")
@login_required 
def dashboard():
    return render_template("dashboard.html", first_name=current_user.first_name, last_name=current_user.last_name, role=current_user.role)

# Dashboard Servies

#Group Meetings
@app.route("/groupmeeting")
def groupmeeting():
    return render_template("groupmeeting.html")
@app.route("/meeting")
@login_required
def meeting():
    return render_template("meeting.html", username=current_user.username)
@app.route("/join", methods=["GET", "POST"])
@login_required
def join():
    if request.method == "POST":
        room_id = request.form.get("roomID")
        return redirect(f"/meeting?roomID={room_id}")

    return render_template("join.html")


#Project Management
# AI Task Prioritization Function
def prioritize_tasks():
    tasks = Project.query.filter(Project.status != 'Completed').all()

    for task in tasks:
        start = datetime.strptime(task.start_date, '%Y-%m-%d')
        due = datetime.strptime(task.deadline, '%Y-%m-%d')
        remaining_days = (due - datetime.today()).days

        # AI logic: Closer deadlines get higher priority
        task.priority = max(1, min(10, 10 - remaining_days))  # Scale between 1-10

    db.session.commit()
@app.route('/projectmanagement')
def projectmanagement():
    projects = Project.query.all()
    return render_template('projectmanagement.html', projects=projects)
@app.route('/add_project', methods=['POST'])
def add_project():
    try:
        data = request.get_json()
        new_project = Project(
            employee_id=data['employee_id'],
            name=data['name'],
            task=data['task'],
            start_date=data['start_date'],
            deadline=data['deadline'],
            status="To Do"
        )
        db.session.add(new_project)
        db.session.commit()
        return jsonify({"message": "Project added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/get_projects', methods=['GET'])
def get_projects():
    projects = Project.query.all()
    project_list = [{
        "id": p.id,
        "employee_id": p.employee_id,
        "name": p.name,
        "task": p.task,
        "start_date": p.start_date,
        "deadline": p.deadline,
        "status": p.status
    } for p in projects]
    
    return jsonify({"projects": project_list})
@app.route('/update_status', methods=['POST'])
def update_status():
    data = request.get_json()
    project = Project.query.get(data['id'])
    
    if project:
        project.status = data['status']
        db.session.commit()
        return jsonify({"message": "Status updated successfully"})
    
    return jsonify({"error": "Project not found"}), 404
# Function to check deadlines
def check_deadlines():
    today = datetime.today().strftime('%Y-%m-%d')
    due_projects = Project.query.filter(Project.deadline == today, Project.status != 'Completed').all()

    for project in due_projects:
        send_email_alert(project)
def send_email_alert(project):
    employee_id = project.employee_id
    if employee_id in employee_emails:
        recipient_email = employee_emails[employee_id]
        subject = f"Project Deadline Alert: {project.name}"
        body = f"Hello,\n\nYour project '{project.name}' with task '{project.task}' is due today ({project.deadline}). Please complete it on time.\n\nBest regards,\nProject Management Team"
        
        msg = Message(subject, recipients=[user.email], body=body)
        mail.send(msg)
        print(f"Email sent to {recipient_email} for project {project.name}.")
# Schedule the deadline checker to run every day at midnight
scheduler = BackgroundScheduler()
scheduler.add_job(check_deadlines, 'cron', hour=0, minute=0)
scheduler.start()


#FusionCode
JD_CLIENT_ID = "2b70d66da397be6c2de4f785fc61e537"
JD_CLIENT_SECRET = "9193f6f99d4346fb52d1d076d01dbe90f8e87f899e813f7ef6d186aa0e4a42d5"
JD_API_URL = "https://api.jdoodle.com/v1/execute"
LANGUAGE_MAP = {
    "python": {"language": "python3", "versionIndex": "3", "ext": ".py"},
    "java": {"language": "java", "versionIndex": "4", "ext": ".java"},
    "cpp": {"language": "cpp14", "versionIndex": "5", "ext": ".cpp"},
    "c": {"language": "c", "versionIndex": "5", "ext": ".c"},
    "javascript": {"language": "nodejs", "versionIndex": "4", "ext": ".js"},
    "typescript": {"language": "typescript", "versionIndex": "4", "ext": ".ts"},
    "go": {"language": "go", "versionIndex": "3", "ext": ".go"},
    "ruby": {"language": "ruby", "versionIndex": "3", "ext": ".rb"},
    "swift": {"language": "swift", "versionIndex": "3", "ext": ".swift"},
    "php": {"language": "php", "versionIndex": "3", "ext": ".php"},
    "rust": {"language": "rust", "versionIndex": "3", "ext": ".rs"},
    "kotlin": {"language": "kotlin", "versionIndex": "3", "ext": ".kt"},
    "dart": {"language": "dart", "versionIndex": "3", "ext": ".dart"},
    "perl": {"language": "perl", "versionIndex": "3", "ext": ".pl"},
    "bash": {"language": "bash", "versionIndex": "3", "ext": ".sh"},
    "r": {"language": "r", "versionIndex": "3", "ext": ".r"}
}
@app.route("/fusioncode")
def fusioncode():
    return render_template("fusioncode.html")
@app.route("/run", methods=["POST"])
def run_code():
    data = request.json
    code = data.get("code", "")
    language = data.get("language", "python")
    user_input = data.get("input", "")
    if language not in LANGUAGE_MAP:
        return jsonify({"error": "Unsupported language"}), 400
    payload = {
        "clientId": JD_CLIENT_ID,
        "clientSecret": JD_CLIENT_SECRET,
        "script": code,
        "language": LANGUAGE_MAP[language]["language"],
        "versionIndex": LANGUAGE_MAP[language]["versionIndex"],
        "stdin": user_input
    }
    response = requests.post(JD_API_URL, json=payload)
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Failed to execute code"}), 500
@app.route("/save_data", methods=["POST"])
def save_code():
    data = request.json
    code = data.get("code", "")
    language = data.get("language", "python")
    if language not in LANGUAGE_MAP:
        return jsonify({"error": "Unsupported language"}), 400
    filename = f"code{LANGUAGE_MAP[language]['ext']}"
    file_path = os.path.join("saved_files", filename)
    os.makedirs("saved_files", exist_ok=True)
    with open(file_path, "w") as f:
        f.write(code)
    return jsonify({"message": "File saved successfully", "filename": filename})
@app.route("/download/<filename>")
def downloadfile(filename):
    file_path = os.path.join("saved_files", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

# Text File Editor
@app.route("/textfileeditor")
@login_required
def textfileeditor():
    files = os.listdir(app.config["UPLOAD_FOLDER"])
    return render_template("textfileeditor.html", files=files)
@app.route("/create", methods=["POST"])
@login_required
def create_file():
    filename = request.form.get("filename", "").strip()
    if filename:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename + ".txt")
        open(filepath, "w").close()
        flash("File created successfully!", "success")
    return redirect(url_for("textfileeditor"))
@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    file = request.files.get("file")
    if file and file.filename.endswith(".txt"):
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))
        flash("File uploaded successfully!", "success")
    else:
        flash("Invalid file format! Only .txt files allowed.", "error")
    return redirect(url_for("textfileeditor"))

@app.route("/edit/<filename>", methods=["GET", "POST"])
@login_required
def edit_file(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if request.method == "POST":
        content = request.form["content"]
        with open(filepath, "w") as f:
            f.write(content)
        flash("File updated successfully!", "success")
        return redirect(url_for("textfileeditor"))
    
    with open(filepath, "r") as f:
        content = f.read()
    return render_template("edit.html", filename=filename, content=content)

@app.route("/save", methods=["POST"])
@login_required
def save_file():
    filename = request.form.get("filename", "").strip()
    content = request.form.get("content", "")

    if not filename:
        return jsonify({'status': 'error', 'message': 'Filename is required'})

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{filename}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return jsonify({'status': 'success', 'message': 'File saved successfully'})

@app.route("/download/<filename>")
@login_required
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

@app.route("/delete/<filename>", methods=["POST"])
@login_required
def delete_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'status': 'success', 'message': 'File deleted successfully'})
    return jsonify({'status': 'error', 'message': 'File not found'})



# File Share Via Mail
# ✅ Allowed File Types (More formats added)
ALLOWED_EXTENSIONS = {
    "pdf", "png", "jpg", "jpeg", "gif", "txt", "doc", "docx", "xls", "xlsx", 
    "csv", "ppt", "pptx", "zip", "rar", "mp4", "mp3"
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/file_share")
def file_share():
    return render_template("file_share.html")

@app.route("/fileshare", methods=["POST"])
def fileshare():
    if "file" not in request.files:
        flash("No file part!", "error")
        return redirect(url_for("file_share"))

    file = request.files["file"]
    recipient_email = request.form["email"]
    custom_message = request.form["message"]

    if file.filename == "":
        flash("No selected file!", "error")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Read file directly into memory (No need to save on disk)
        file_data = file.read()

        # Sending Email
        msg = Message("📎 File from Flask App", recipients=[recipient_email])
        msg.body = custom_message
        msg.attach(filename, "application/octet-stream", file_data)

        try:
            mail.send(msg)
            flash("✅ File sent successfully!", "success")
        except Exception as e:
            flash(f"❌ Error sending email: {str(e)}", "error")

        return redirect(url_for("file_share"))
    else:
        flash("❌ Invalid file type!", "error")
        return redirect(url_for("file_share"))


# Project Planner
@app.route('/planner')
def planner():
    return render_template('project_planner.html')

@app.route('/generate_roadmap', methods=['POST'])
def generate_roadmap():
    data = request.json
    topic = data.get("topic", "")
    
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(
        f"Create a structured roadmap for a project on '{topic}'. Include:\n"
        "- Key Phases (e.g., Planning, Development, Testing, Deployment)\n"
        "- Important Milestones and Deadlines\n"
        "- Task Breakdown with estimated durations\n"
        "- Dependencies (which tasks need to be completed first)\n"
    )

    roadmap_text = response.text

    # Generate PDF
    pdf_path = "roadmap.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont("Helvetica", 12)
    
    y_position = 800
    for line in roadmap_text.split("\n"):
        c.drawString(50, y_position, line)
        y_position -= 20  # Move down for next line

        if y_position < 50:  # If reaching page bottom, create new page
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 800

    c.save()

    return jsonify({"roadmap": roadmap_text, "pdf": f"/download_pdf"})

@app.route('/download_pdf')
def download_pdf():
    return send_file("roadmap.pdf", as_attachment=False)

#poll System
@app.route('/pollhome')
def pollhome():
    polls = Poll.query.all()
    return render_template('pollhome.html', polls=polls)

@app.route('/poll/<int:poll_id>')
def poll(poll_id):
    poll = Poll.query.get_or_404(poll_id)
    options = Option.query.filter_by(poll_id=poll_id).all()
    return render_template('poll.html', poll=poll, options=options)

@app.route('/add_poll', methods=['POST'])
def add_poll():
    question = request.form.get('question')
    options = request.form.get('options').split(',')
    
    if question and options:
        new_poll = Poll(question=question)
        db.session.add(new_poll)
        db.session.commit()

        for option in options:
            new_option = Option(poll_id=new_poll.id, text=option.strip())
            db.session.add(new_option)

        db.session.commit()
    
    return redirect(url_for('pollhome'))
from flask_login import current_user, login_required

@app.route('/delete_poll/<int:poll_id>', methods=['POST'])
@login_required
def delete_poll(poll_id):
    # Check if the logged-in user is an admin
    if current_user.role != "admin":
        return "Unauthorized Access!", 403  # Return forbidden error

    poll = Poll.query.get_or_404(poll_id)
    
    # Delete all options related to this poll
    Option.query.filter_by(poll_id=poll.id).delete()

    # Delete all votes related to this poll
    Vote.query.filter_by(poll_id=poll.id).delete()

    # Delete the poll itself
    db.session.delete(poll)
    db.session.commit()

    return redirect(url_for('pollhome'))

@app.route('/vote/<int:poll_id>', methods=['POST'])
def vote(poll_id):
    if 'user_id' not in session:
        session['user_id'] = request.remote_addr  # Simulating user tracking

    user_id = session['user_id']
    
    # Check if user has already voted
    existing_vote = Vote.query.filter_by(poll_id=poll_id, user_id=user_id).first()
    if existing_vote:
        return "You have already voted in this poll!", 403
    
    option_id = request.form.get('option')
    selected_option = Option.query.get(option_id)
    
    if selected_option:
        selected_option.votes += 1
        db.session.add(Vote(poll_id=poll_id, user_id=user_id))
        db.session.commit()

    return redirect(url_for('poll', poll_id=poll_id))


#Content Summerizer
@app.route('/content')
def content():
    return render_template('sum.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    content = data.get("content", "")
    
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(
        f"Summarize the following text concisely:\n{content}"
    )

    summary_text = response.text
    
    return jsonify({"summary": summary_text})


#Timer
active_timers = {}

@app.route('/timer')
def timer():
    return render_template('timer.html')

@app.route('/start_timer', methods=['POST'])
def start_timer():
    task = request.json.get('task')
    if task not in active_timers:
        active_timers[task] = time.time()
    return jsonify({"message": "Timer started", "task": task})

@app.route('/stop_timer', methods=['POST'])
def stop_timer():
    task = request.json.get('task')
    if task in active_timers:
        elapsed_time = round(time.time() - active_timers.pop(task))
        return jsonify({"message": "Timer stopped", "task": task, "elapsed_time": elapsed_time})
    return jsonify({"error": "No active timer found for task"}), 400

@app.route('/save_record', methods=['POST'])
def save_record():
    data = request.json
    username = data.get('username')
    task = data.get('task')
    duration = data.get('duration')

    if not username or not task or not duration:
        return jsonify({"error": "Missing data"}), 400

    new_record = TaskRecord(username=username, task=task, duration=duration)
    db.session.add(new_record)
    db.session.commit()

    return jsonify({"message": "Record saved successfully"})

@app.route('/get_records', methods=['GET'])
def get_records():
    records = TaskRecord.query.all()
    records_list = [{"username": r.username, "task": r.task, "duration": r.duration} for r in records]
    return jsonify(records_list)

@app.route("/attendance", methods=["GET", "POST"])
@login_required
def attendance():
    search_query = request.form.get("search", "").strip()
    
    if search_query:
        attendance_records = Attendance.query.join(Register).filter(Register.username.ilike(f"%{search_query}%")).order_by(Attendance.date.desc()).all()
    else:
        attendance_records = Attendance.query.order_by(Attendance.date.desc()).all()

    return render_template("attendance.html", attendance_records=attendance_records, search_query=search_query)



# Intelligent Resource Finder
GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_PAT = os.getenv("GITHUB_PAT")

def fetch_github_repos(query):
    """Fetch top GitHub repositories based on query."""
    params = {"q": query, "sort": "stars", "order": "desc"}
    headers = {"Accept": "application/vnd.github.v3+json"}

    if GITHUB_PAT:  # Use token if available
        headers["Authorization"] = f"token {GITHUB_PAT}"

    response = requests.get(GITHUB_API_URL, params=params, headers=headers)

    if response.status_code != 200:
        return [("GitHub API error. Check token or rate limits.", "#")]

    return [(repo["name"], repo["html_url"]) for repo in response.json().get("items", [])]

def fetch_kaggle_resources(query):
    """Use Gemini AI to generate Kaggle datasets, competitions, and notebooks in structured JSON format."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Find the top Kaggle datasets, competitions, and notebooks related to '{query}'.
        Provide results in JSON format:
        {{
          "datasets": [{{ "title": "Dataset Name", "url": "Dataset URL" }}],
          "competitions": [{{ "title": "Competition Name", "url": "Competition URL" }}],
          "notebooks": [{{ "title": "Notebook Title", "url": "Notebook URL" }}]
        }}
        Ensure the response is strictly in JSON format with no additional text.
        """

        response = model.generate_content(prompt)

        # Extract JSON using regex (fallback)
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if json_match:
            kaggle_data = json.loads(json_match.group(0))
            return kaggle_data

        return {"error": "Invalid JSON response from Gemini AI."}

    except json.JSONDecodeError:
        return {"error": "Gemini AI returned non-JSON output."}

    except Exception as e:
        return {"error": f"Gemini AI Error: {str(e)}"}

def generate_summary(text):
    """Generate an AI-powered summary."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(f"Provide a 4-line summary: {text}")
        return response.text if response else "No summary available."
    except Exception as e:
        return "Summary generation failed."

@app.route("/resource")
def resource():
    return render_template("resource.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({
            "error": "Query cannot be empty."
        }), 400

    # 1️⃣ GitHub Repositories
    github_results = fetch_github_repos(query)

    # 2️⃣ Kaggle Resources (Datasets, Competitions, Notebooks)
    kaggle_results = fetch_kaggle_resources(query)

    # 3️⃣ Gemini AI Summary
    summary = generate_summary(f"Find resources on: {query}")

    # 4️⃣ Internet Archive
    archive_output = []
    try:
        archive_url = f"https://archive.org/advancedsearch.php?q={query}&fl[]=identifier,title,creator&rows=5&output=json"
        archive_res = requests.get(archive_url)
        archive_res.raise_for_status()
        docs = archive_res.json().get('response', {}).get('docs', [])
        for doc in docs:
            title = doc.get('title', 'Untitled')
            creator = doc.get('creator', 'Unknown')
            identifier = doc.get('identifier', '')
            link = f"https://archive.org/details/{identifier}"
            archive_output.append({
                "title": title,
                "author": creator,
                "url": link
            })
    except Exception as e:
        archive_output.append({"error": str(e)})

    # 5️⃣ OpenLibrary
    openlib_output = []
    try:
        ol_res = requests.get("https://openlibrary.org/search.json", params={"q": query})
        ol_res.raise_for_status()
        books = ol_res.json().get("docs", [])[:5]
        for book in books:
            title = book.get("title", "Untitled")
            authors = ", ".join(book.get("author_name", ["Unknown"]))
            olid = book.get("key", "")
            link = f"https://openlibrary.org{olid}"
            openlib_output.append({
                "title": title,
                "author": authors,
                "url": link
            })
    except Exception as e:
        openlib_output.append({"error": str(e)})

    # 6️⃣ arXiv API
    arxiv_output = []
    try:
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
        arxiv_res = requests.get(arxiv_url)
        arxiv_res.raise_for_status()
        entries = arxiv_res.text.split("<entry>")
        for entry in entries[1:]:
            try:
                title = entry.split("<title>")[1].split("</title>")[0].strip().replace("\n", "")
                link = entry.split("<id>")[1].split("</id>")[0].strip()
                arxiv_output.append({
                    "title": title,
                    "url": link
                })
            except:
                continue
    except Exception as e:
        arxiv_output.append({"error": str(e)})

    # 📦 Final JSON Response
    return jsonify({
        "github": github_results,
        "kaggle": kaggle_results,
        "summary": summary,
        "archive": archive_output,
        "openlibrary": openlib_output,
        "arxiv": arxiv_output
    })





@app.route('/whiteboard')
def whiteboard():
    return render_template('whiteboard.html')

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)

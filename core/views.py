from django.shortcuts import render, redirect
from processing.services import TextProcessingService

def home(request):
    return render(request, 'core/home.html')


from django.shortcuts import render, redirect
from processing.services import TextProcessingService


def home(request):
    if request.method == "POST":
        input_text = request.POST.get("input_text")

        if input_text and len(input_text) >= 100:
            service = TextProcessingService(input_text)
            result_data = service.process()

            request.session["result"] = result_data
        else:
            request.session["result"] = {
                "detection": {
                    "label": "Input too short (minimum 100 characters required)",
                    "confidence": "Low confidence"
                }
            }

        return redirect("home")

    result = request.session.pop("result", None)

    return render(request, 'home.html', {"result": result})




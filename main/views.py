import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from rouge import Rouge
from datasets import load_metric

# Create your views here.


def index(request):
    return render(request, 'main/index.html')


def demo(request):
    return render(request, f'main/demo.html')


rouge = Rouge()
bertscore_metric = load_metric('bertscore')

@csrf_exempt
def submit(request):
    jsonObject = json.loads(request.body)

    reference = jsonObject.get('gold')
    answer = jsonObject.get('answer')
    score = rouge.get_scores(answer, reference)
    rouge_l_f1 = score[0]["rouge-l"]["f"]
    bert_score = bertscore_metric.compute(predictions=[answer], references=[reference], lang="en")
    bertscore_f1 = bert_score["f1"][0]

    response = {"rouge_score": round(rouge_l_f1*100), "bert_score": round(bertscore_f1*100)}
    return JsonResponse(response)
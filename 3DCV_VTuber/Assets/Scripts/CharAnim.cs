using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CharAnim : MonoBehaviour
{
    private Animator charAnim;
    private GameObject selfChar;

    private Live2D.Cubism.Framework.Expression.CubismExpressionController expressionControl;

    private void Start()
    {
        charAnim = GetComponent<Animator>();
        expressionControl = GetComponent<Live2D.Cubism.Framework.Expression.CubismExpressionController>();
    }

    private void Update()
    {
        CharacterMotion();
        FacialExpressions();
    }

    void CharacterMotion()
    {
        if(Input.GetKeyDown(KeyCode.Q))
        {
            charAnim.SetTrigger("madTrigger");
        }
        else if(Input.GetKeyDown(KeyCode.W))
        {
            charAnim.SetTrigger("embarassedTrigger");
        }
    }

    void FacialExpressions()
    {

        if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            expressionControl.CurrentExpressionIndex = 0;
        } 
        else if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            expressionControl.CurrentExpressionIndex = 1;
        }
        else if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            expressionControl.CurrentExpressionIndex = 2;
        }
        else if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            expressionControl.CurrentExpressionIndex = 3;
        }
        else if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            expressionControl.CurrentExpressionIndex = 4;
        }
        else if (Input.GetKeyDown(KeyCode.Alpha5))
        {
            expressionControl.CurrentExpressionIndex = 5;
        }
        else if (Input.GetKeyDown(KeyCode.Alpha6))
        {
            expressionControl.CurrentExpressionIndex = 6;
        }
        else if (Input.GetKeyDown(KeyCode.Alpha7))
        {
            expressionControl.CurrentExpressionIndex = 7;
        }


    }
}
